# @Time   : 2025/09
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it
# source: https://github.com/justinhangoebl/Semantic-ID-Generation

import argparse
import os
from typing import NamedTuple

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import rich


class RQ_VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims,
        codebook_size,
        codebook_kmeans_init=True,
        codebook_sim_vq=True,
        n_quantization_layers=3,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.codebook_kmeans_init = codebook_kmeans_init
        self.codebook_sim_vq = codebook_sim_vq
        self.commitment_weight = commitment_weight

        self.quantization_layers = nn.ModuleList(
            modules=[
                Quantization(
                    latent_dim=latent_dim,
                    codebook_size=codebook_size,
                    commitment_weight=commitment_weight,
                    do_kmeans_init=codebook_kmeans_init,
                    sim_vq=codebook_sim_vq,
                )
                for _ in range(n_quantization_layers)
            ]
        )

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dims=hidden_dims[::-1],
            latent_dim=latent_dim,
        )

    @torch.no_grad()
    def kmeans_init_codebooks(self, data):
        """
        Initializes all quantization layers using k-means with full (or large) dataset.
        Call this before training.
        """
        x = self.encoder(data.to(self.device).float())
        for layer in self.quantization_layers:
            layer._kmeans_init(x)
            emb = layer.get_item_embeddings(layer(x).ids)
            x = x - emb

    def get_semantic_ids(self, x):
        res = self.encoder(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(res)
            quantized = layer(res)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb  # Update residuals
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
        )

    def forward(self, x):
        quantized = self.get_semantic_ids(x)
        embs = quantized.embeddings
        x_hat = self.decoder(embs.sum(axis=-1))
        x_hat = torch.nn.functional.normalize(x_hat, p=2)

        # Using sum as the loss to match the previous behavior
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(  # noqa: E501
                        axis=-1
                    ),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLossesOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )


class Quantization(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity()

    @torch.no_grad
    def _kmeans_init(self, x: Tensor):
        x = x.view(-1, self.embed_dim).cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, n_init=10)
        kmeans.fit(x)
        self.embedding.weight.copy_(torch.from_numpy(kmeans.cluster_centers_))
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids):
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x):
        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        codebook = self.out_proj(self.embedding.weight)

        dist = (x**2).sum(axis=1, keepdim=True) + (codebook.T**2).sum(axis=0, keepdim=True) - 2 * x @ codebook.T

        _, ids = (dist.detach()).min(axis=1)
        emb = self.get_item_embeddings(ids)
        emb_out = x + (emb - x).detach()

        # Compute commitment loss
        emb_loss = ((x.detach() - emb) ** 2).sum(axis=[-1])
        query_loss = ((x - emb.detach()) ** 2).sum(axis=[-1])
        loss = emb_loss + self.commitment_weight * query_loss

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss,
        )


class Encoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256], latent_dim=128):
        super().__init__()
        layers = []

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())  # ReLU in the paper
        layers.append(nn.Linear(dims[-1], latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, output_dim=768, hidden_dims=[256, 512], latent_dim=128):
        super().__init__()
        layers = []
        dims = [latent_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.network(z))


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLossesOutput(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset Name.")
    parser.add_argument("--data_path", type=str, required=True, help="Where the dataset is stored.")
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--weight_decay", type=int, default=1e-4)
    parser.add_argument("--learning_rate", type=int, default=1e-3)

    parser.add_argument("--gpu_id", type=int, default=0)
    # Sentence Encoder Weight Size
    parser.add_argument("--input_dimension", type=int, default=768)

    # RQ-VAE Parameters
    parser.add_argument("--hidden_dimensions", type=list, default=[256, 128, 64])
    parser.add_argument("--latent_dimension", type=int, default=256)
    parser.add_argument("--num_codebook_layers", type=int, default=3)
    parser.add_argument("--codebook_clusters", type=int, default=256)
    parser.add_argument("--commitment_weight", type=float, default=0.25)

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.gpu_id}")
    return args


def get_dataset_df(args):
    return pd.read_csv(os.path.join(args.data_path, args.dataset, f"{args.dataset}.item"), sep=args.sep)


def get_item_sentences(args):
    dataset_df = get_dataset_df(args)
    headers = dataset_df.columns.tolist()
    # rename each column by taking everything before ":"
    dataset_df.columns = [header.split(":")[0] for header in headers]

    # create for each row a new column with the sentence <column_name>:value
    prompts = []
    for _, row in dataset_df.iterrows():
        prompt = " ".join([f"{header}: {row[header]}" for header in dataset_df.columns])
        prompts.append(prompt)

    return dataset_df.item_id, prompts


def get_rq_vae_model(args):
    rq_vae_model = RQ_VAE(
        input_dim=args.input_dimension,
        latent_dim=args.latent_dimension,
        hidden_dims=args.hidden_dimensions,
        codebook_size=args.codebook_clusters,
        codebook_kmeans_init=True,
        codebook_sim_vq=True,
        n_quantization_layers=args.num_codebook_layers,
        commitment_weight=args.commitment_weight,
    ).to(args.device)

    return rq_vae_model


def train(args, model, data):
    results = []
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    kmeans_init_data = torch.Tensor(data[torch.arange(min(20000, len(data)))]).to(args.device)
    model(kmeans_init_data)

    data = DataLoader(data, batch_size=args.batch_size)
    for epoch in rich.tqdm(range(args.epochs), total=args.epochs, desc="Training Loop"):
        total_loss = 0
        total_reconstruction_loss = 0
        total_commit_loss = 0
        p_unique = 0

        for batch in data:
            batch = batch.to(args.device).float()  # noqa: PLW2901
            optimizer.zero_grad()
            result = model(batch)
            result.loss.backward()
            optimizer.step()

            total_loss += result.loss.item()
            total_reconstruction_loss += result.reconstruction_loss.item()
            total_commit_loss += result.rqvae_loss.item()
            p_unique += result.p_unique_ids.item()

        epoch_stats = {
            "Epoch": epoch,
            "Loss": total_loss / len(data),
            "Reconstruction Loss": total_reconstruction_loss / len(data),
            "RQ-VAE Loss": total_commit_loss / len(data),
            "Prob Unique IDs": p_unique / len(data),
        }
        if epoch % args.print_every == 0:
            print(epoch_stats)

        results.append(epoch_stats)
    return results


if __name__ == "__main__":
    args = parse_args()
    sentence_transformer = SentenceTransformer("sentence-transformers/sentence-t5-xl", device=args.device)
    items, items_sentences = get_item_sentences(args)
    encoded_sentences = sentence_transformer.encode(
        sentences=items_sentences, show_progress_bar=True, convert_to_tensor=True
    )

    rq_vae_model = get_rq_vae_model(args)
    results = train(args, rq_vae_model, encoded_sentences)

    # generate final semantic embeddings
    rq_vae_output = rq_vae_model.get_semantic_ids(encoded_sentences)
    semantic_ids = rq_vae_output.sem_ids
    semantic_ids_embs = rq_vae_output.embeddings

    # handle collisions
    mappings = dict(
        items=items.tolist(),
        semantic_ids=semantic_ids.cpu().numpy().tolist(),
        semantic_ids_embs=semantic_ids_embs.detach().cpu().numpy().tolist(),
    )

    # Save the mappings to a CSV file
    pd.DataFrame(mappings).to_csv(
        os.path.join(args.data_path, args.dataset, f"{args.dataset}.semanticids"), index=False
    )
