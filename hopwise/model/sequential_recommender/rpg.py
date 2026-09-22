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

from hopwise.model.abstract_recommender import SequentialRecommender
from hopwise.model.layers import ResidualBlock


class RPG(SequentialRecommender):
    r"""RPG is a recommendation model that generates each token of the next semantic ID in parallel."""

    def __init__(self, config, dataset):
        from transformers import GPT2Config, GPT2Model

        super().__init__(config, dataset)

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

        if self.use_gcd:
            # the neighbors of the best beam are n_edges distinct items, so there are always enough beams to select
            if self.n_beams > self.n_edges:
                raise ValueError(f"n_beams [{self.n_beams}] should not be greater than n_edges [{self.n_edges}].")
            if self.n_edges > self.n_items - 1:
                raise ValueError(f"n_edges [{self.n_edges}] should not be greater than the number of items.")

        self.item2shifted_sem_id = dataset.item2shifted_sem_id.to(self.device)
        self.n_digit = dataset.n_digit
        self.codebook_size = dataset.codebook_size

        gpt2config = GPT2Config(
            vocab_size=dataset.vocab_size,
            n_positions=config["MAX_ITEM_LIST_LENGTH"],
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

        # item-item graph used for graph-constrained decoding, built once per evaluation
        self.adjacency = None

    def train(self, mode=True):
        # model weights change during training, so the decoding graph must be rebuilt at the next evaluation
        if mode:
            self.adjacency = None
        return super().train(mode)

    def forward(self, item_seq):
        input_tokens = self.item2shifted_sem_id[item_seq]
        attention_mask = (item_seq != 0).long()
        # aggregate semantic ids embeddings averaging embeddings for each item
        wte = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(inputs_embeds=wte, attention_mask=attention_mask)
        # outputs.last_hidden_state: shape (bs, seq_len(50), embedding_size)
        heads_final_states = [
            self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)
        ]  # bs, 50, 1, 448
        heads_final_states = torch.cat(heads_final_states, dim=-2)  # bs,50,32,448
        return heads_final_states

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        # Each augmented sequence is supervised only on its target item, predicted from the last position.
        # It is equivalent to the original implementation, which supervises every position of the first
        # max_seq_length items at once and only the last position of the following sliding windows.
        # shape: (bs, seq_len, n_pred_head (semantic id size), embedding_size)
        hidden_states = self.forward(item_seq)
        selected_states = hidden_states.gather(
            dim=1, index=(item_seq_len - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.embedding_size)
        ).squeeze(1)  # shape: (bs, n_pred_head, embedding_size)
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
        token_emb = self.gpt2.wte.weight[1:-1]  # vocab_size, emb_size -> 8192, 448
        token_emb = F.normalize(token_emb, dim=-1)

        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        # calculate the output of each head
        token_logits = [
            torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
            for i in range(self.n_pred_head)
        ]
        # convert each item to the corresponding semantic id
        token_labels = self.item2shifted_sem_id[pos_items]

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
        return scores.gather(dim=1, index=test_item.unsqueeze(1)).squeeze(1)

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
        batch_size = token_logits.shape[0]

        if self.adjacency is None:
            self.adjacency = self.init_graph()
        adjacency = self.adjacency

        results = torch.full((batch_size, self.n_items), -torch.inf, device=self.device)

        # Randomly sample n_beams item ids in [1, n_items) as initial beams
        topk_nodes_sorted = torch.randint(
            1, self.n_items, (batch_size, self.n_beams), dtype=torch.long, device=token_logits.device
        )

        for propagation_step in range(self.propagation_steps):
            # Find the neighbors of the current beams. The adjacency list is indexed by item id
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
                    topk = torch.topk(scores, min(max(self.topk), scores.shape[0])).indices
                    results[batch_id, neighbors_in_batch[topk]] = scores[topk]
                else:
                    # otherwise, select beams and propagate again
                    topk = torch.topk(scores, self.n_beams).indices

                next_nodes.append(neighbors_in_batch[topk])

            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        return results

    @torch.no_grad()
    def init_graph(self):
        """Builds the item-item graph used for graph-constrained decoding.

        The similarity of two items is the average over the digits of the cosine similarity, rescaled in [0, 1],
        between the token embeddings of their semantic IDs. Each item is connected to its ``n_edges`` most similar
        items. Similarities are computed in chunks of ``chunk_size`` items, such that the full ``n_items x n_items``
        similarity matrix is never materialized.

        Returns:
            torch.Tensor: The adjacency list of shape ``[n_items, n_edges]``. Row 0 (PAD) is not used.
        """
        device = self.gpt2.wte.weight.device

        # token embeddings of each digit, ignoring PAD and EOS tokens. shape: (n_digit, codebook_size, d)
        wte = F.normalize(self.gpt2.wte.weight[1:-1].view(self.n_digit, self.codebook_size, -1), dim=-1)
        # pairwise similarities between the codewords of each digit, from [-1, 1] to [0, 1]
        # shape: (n_digit, codebook_size, codebook_size)
        token_sims = 0.5 * (torch.bmm(wte, wte.transpose(1, 2)) + 1.0)

        # codeword index of each digit for each item, excluding PAD. shape: (n_items - 1, n_digit)
        digit_offsets = torch.arange(self.n_digit, device=device) * self.codebook_size + 1
        codes = self.item2shifted_sem_id[1:].to(device) - digit_offsets

        adjacency = torch.zeros((self.n_items, self.n_edges), dtype=torch.long, device=device)
        for i_start in range(0, codes.shape[0], self.chunk_size):
            codes_i = codes[i_start : i_start + self.chunk_size]

            # average similarity between the items of the chunk and all the items. shape: (chunk_size, n_items - 1)
            item_sims = torch.zeros((codes_i.shape[0], codes.shape[0]), device=device)
            for k in range(self.n_digit):
                item_sims += token_sims[k].index_select(0, codes_i[:, k]).index_select(1, codes[:, k])
            item_sims /= self.n_digit

            # column indices are shifted by 1 to obtain item ids, so PAD can never be a neighbor
            adjacency[i_start + 1 : i_start + 1 + codes_i.shape[0]] = torch.topk(item_sims, k=self.n_edges).indices + 1

        return adjacency
