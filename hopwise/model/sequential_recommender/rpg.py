# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# UPDATE
# @Time   : 2025/02/19
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

r"""RPG
################################################
    Reference:
    Hou Yupeng et al. "Generating Long Semantic IDs in Parallel for Recommendation".
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from hopwise.model.abstract_recommender import SequentialRecommender
from hopwise.model.layers import ResidualBlock


class RPG(SequentialRecommender):
    r"""RPG is a recommendation model that generates each token of the next semantic ID in parallel."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        dataset.item2shifted_sem_id = dataset.item2shifted_sem_id.to(self.device)

        self.topk = config["topk"]
        self.use_gcd = config["use_gcd"]
        self.codebook_size = config["codebook_size"]
        self.embedding_size = config["embedding_size"]
        self.temperature = config["temperature"]
        self.chunk_size = config["chunk_size"]
        self.n_edges = config["n_edges"]
        self.n_beams = config["n_beams"]
        self.propagation_steps = config["propagation_steps"]
        self.loss_type = config["loss_type"]  # necessary otherwise hopwise don't recognize seq recommender

        self.item2shifted_sem_id = dataset.item2shifted_sem_id
        self.n_digit = dataset.n_digit
        self.codebook_size = dataset.codebook_size

        gpt2config = GPT2Config(
            vocab_size=dataset.vocab_size,
            n_positions=config["max_seq_length"],
            n_embd=config["embedding_size"],
            n_layer=config["layers"],
            n_head=config["heads"],
            n_inner=config["embedding_size_inner_mlp"],
            activation_function=config["activation_function"],
            resid_pdrop=config["resid_pdrop"],
            embd_pdrop=config["embd_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            initializer_range=config["initializer_range"],
            eos_token_id=dataset.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        # Number of values in a semantic id.
        self.n_pred_head = dataset.n_digit
        pred_head_list = []

        # Create prediction heads with a Residual Connection
        for _ in range(self.n_pred_head):
            pred_head_list.append(ResidualBlock(config["embedding_size"]))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, item_seq):
        input_tokens = self.item2shifted_sem_id[item_seq]
        attention_mask = (item_seq != 0).long()
        # aggregate semantic ids embeddings averaging embeddings for each item
        wte = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(inputs_embeds=wte, attention_mask=attention_mask)
        heads_final_states = [
            self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)
        ]
        heads_final_states = torch.cat(heads_final_states, dim=-2)

        return heads_final_states

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        padding = torch.zeros((item_seq.size(0), 1), dtype=item_seq.dtype, device=item_seq.device)

        labels = torch.cat([item_seq[:, :-1], padding], dim=1)
        item_seq = torch.cat([item_seq[:, 1:], padding], dim=1)

        # Calculate representation
        hidden_states = self.forward(
            item_seq
        )  # shape: (bs, seq_len, n_pred_head (semantic id size), embedding_size). (2048, 50,4,448)
        label_mask = labels.view(-1) != 0
        selected_states = hidden_states.view(-1, self.n_pred_head, self.embedding_size)[label_mask]
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)

        # split the embedding into n_pred_head parts along dimension 0
        token_embs = torch.chunk(
            token_emb, self.n_pred_head, dim=0
        )  # tupla di 4 elementi. Ogni elemento è un tensore di size (256, 448)
        # calculate the output of each head
        token_logits = [
            torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
            for i in range(self.n_pred_head)
        ]
        # convert each item to the corresponding semantic id
        token_labels = self.item2shifted_sem_id[labels.view(-1)[label_mask]]

        # aggregate loss over the prediction heads
        losses = [
            self.loss(token_logits[i], token_labels[:, i] - i * self.codebook_size - 1)
            for i in range(self.n_pred_head)
        ]
        loss = torch.mean(torch.stack(losses))
        return loss

    def predict(self, interaction):
        """Predict scores for the next item in the sequence. Used only in GFLOPS fn"""
        test_item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)
        return scores[:, test_item]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        hidden_states = self.forward(item_seq)
        hidden_states = hidden_states.gather(
            dim=1, index=(item_seq_len - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.embedding_size)
        )
        hidden_states = F.normalize(hidden_states, dim=-1)

        # Do not consider PAD token and EOS token.
        token_emb = self.gpt2.wte.weight[1:-1]

        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [
            torch.matmul(hidden_states[:, 0, i, :], token_embs[i].T) / self.temperature
            for i in range(self.n_pred_head)
        ]
        # create probability distribution
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)  # (batch_size, n_tokens)

        if self.use_gcd:
            scores = self.graph_propagation(token_logits=token_logits)
        else:
            scores = torch.gather(
                # (batch_size, n_items, n_tokens)
                input=token_logits.unsqueeze(-2).expand(-1, self.n_items, -1),
                dim=-1,
                # (batch_size, n_items, code_dim)
                index=(self.item2shifted_sem_id[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
            ).mean(dim=-1)

            # account for PAD
            padding = torch.full((item_seq.size(0), 1), -torch.inf, device=item_seq.device)
            scores = torch.cat([padding, scores], dim=1)

        return scores

    def graph_propagation(self, token_logits):
        """
        I don't like this implementation. It propagates only on a random subset
        (#self.n_beams) of nodes. Does it make sense?
        """

        batch_size = token_logits.shape[0]

        adjacency = self.init_graph()

        results = torch.full((batch_size, self.n_items), -torch.inf, device=self.device)

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.n_items, (batch_size, self.n_beams), dtype=torch.long, device=token_logits.device
        )

        for propagation_step in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                # scores for neighbors
                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item2shifted_sem_id[neighbors_in_batch] - 1),
                ).mean(dim=-1)

                # if it's the last propagation step, save the scores
                if propagation_step == self.propagation_steps - 1:
                    topk = torch.topk(scores, max(self.topk)).indices
                    results[batch_id, topk] = scores[topk]
                else:
                    # otherwise, select beams and propagate again
                    topk = torch.topk(scores, self.n_beams).indices

                next_nodes.append(neighbors_in_batch[topk])

            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        return results

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)

        wte = self.gpt2.wte.weight[1:-1].view(self.n_digit, self.codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        wte = F.normalize(wte, dim=-1)
        token_sims = torch.bmm(wte, wte.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((self.n_items, self.n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, self.n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, self.n_items)

            # shape: (chunk_i_size, 32)
            # sub-block for items i
            tokens_i = self.item2shifted_sem_id[i_start:i_end]

            for j_start in range(1, self.n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, self.n_items)

                # shape: (chunk_j_size, 32)
                # sub-block for items j
                tokens_j = self.item2shifted_sem_id[j_start:j_end]

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(self.n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * self.codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * self.codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / self.n_digit

                # Write back into the final item_item_sim
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def init_graph(self):
        item_item_sim = self.build_ii_sim_mat()
        adjacency = torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices
        return adjacency
