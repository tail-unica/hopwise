# @Time   : 2025/05/25
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it


import math
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.trainer.hf_path_trainer import ConstrainedLogitsProcessorWordLevel, get_tokenized_used_ids
from hopwise.utils import InputType, ModelType, PathLanguageModelingTokenType

TokenType = IntEnum("TokenType", [("SPECIAL", 0), ("USER", 1), ("ENTITY", 2), ("RELATION", 3)])


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AutoregressiveSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["n_heads"]
        self.dropout = config["dropout"]
        # Reduce the projection dim to match desired output dim
        self.head_dim = config["hidden_size"] // config["n_heads"]

        assert config["hidden_size"] % config["n_heads"] == 0

        # the second hidden size could be different
        self.W_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # regularization
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.causal_mask = (
            torch.triu(torch.ones(config["context_length"], config["context_length"]), diagonal=1)
            .bool()
            .to(config["device"])
        )

    def forward(self, x):
        # B,C,T = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        batch_size, seq_length, hidden_size = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # applies a linear transformation on the last dimension: xW^T = (9,256)(256,256)
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        k = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # 4096,4,9,64
        v = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        q = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Dot product for each head
        # Calculate attention scores
        attn_scores = (q @ k.transpose(2, 3)) / (1.0 / math.sqrt(k.size(-1)))

        # truncating the mask to the current sequence length
        causal_mask = self.causal_mask.bool()[:seq_length, :seq_length]
        # apply causal masking
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

        # calculate attention scores probabilities
        attn_scores = F.softmax(attn_scores, dim=-1)

        attn_scores = self.attn_dropout(attn_scores)

        context_vec = (attn_scores @ v).transpose(1, 2)

        context_vec = context_vec.reshape(batch_size, seq_length, self.hidden_size)
        context_vec = self.out_proj(context_vec)  # optional projection
        context_vec = self.resid_dropout(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["hidden_size"], 4 * config["hidden_size"], bias=config["bias"])
        self.silu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["hidden_size"], config["hidden_size"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_1 = LayerNorm(config["hidden_size"], bias=config["bias"])
        self.causal_attn = AutoregressiveSelfAttention(config)
        self.layernorm_2 = LayerNorm(config["hidden_size"], bias=config["bias"])
        self.feedforward = FeedForward(config)

    def forward(self, x):
        x = self.layernorm_1(x)
        x = x + self.causal_attn(x)
        x = self.layernorm_2(x)
        x = x + self.feedforward(x)

        return x


class PEARLMgpt2(KnowledgeRecommender):
    """
    Reference:
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
    """

    input_type = InputType.PATHWISE
    type = ModelType.PATH_LANGUAGE_MODELING

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config
        self.dataset = dataset

        self.tokenizer = dataset.tokenizer

        self.used_ids = dataset._used_ids
        self.tokenized_ckg = dataset.get_tokenized_ckg()
        self.tokenized_used_ids = get_tokenized_used_ids(self.used_ids, self.tokenizer)
        self.n_users = dataset.user_num
        self.n_entities = dataset.entity_num
        self.n_items = dataset.item_num
        self.n_relations = dataset.relation_num
        self.temperature = config["temperature"]
        self.path_hop_length = config["path_hop_length"]
        self.path_gen_args = self.config["path_generation_args"]

        # path_hop_length = n_relations => (n_relations + user_starting_node) + n_relations + 2 (BOS, EOS)
        self.token_sequence_length = 1 + self.path_hop_length + self.path_hop_length + 1
        self.ranker_max_new_tokens = self.token_sequence_length - 3

        self.wte = nn.Embedding(len(self.tokenizer), config["hidden_size"])
        self.wpe = nn.Embedding(config["context_length"], config["hidden_size"])
        self.wp_type_e = nn.Embedding(len(TokenType), config["hidden_size"])

        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layers"])])
        self.layernorm = nn.LayerNorm(config["hidden_size"], bias=config["bias"])

        self.lm_head = nn.Linear(config["hidden_size"], len(self.tokenizer), bias=False)

        # weight tying
        self.wte.weight = self.lm_head.weight

        self.logit_processor = ConstrainedLogitsProcessorWordLevel(
            self.tokenized_ckg,
            self.tokenized_used_ids,
            self.token_sequence_length,
            self.tokenizer,
            self.path_gen_args["paths_per_user"],
            task=ConstrainedLogitsProcessorWordLevel.RECOMMENDATION_TASK,
        )

        self.loss = nn.CrossEntropyLoss()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layers"]))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        bs, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # position embeddings of shape (t, n_embd)
        type_emb_pos = torch.tensor(
            # BOS + USER + REL + ENT + REL + ... + ENT + REL + EOS
            [TokenType.SPECIAL.value, TokenType.USER.value]
            + [TokenType.RELATION.value, TokenType.ENTITY.value] * self.config["path_hop_length"]
            + [TokenType.SPECIAL.value]
        ).to(self.device)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)[: tok_emb.size(1)]
        type_emb = self.wp_type_e(type_emb_pos)[: tok_emb.size(1)]

        # think about get_flops, it restrict the max length of the input
        # x = tok_emb + pos_emb + type_emb
        x = tok_emb + pos_emb + type_emb
        # x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        # x = self.layernorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits[:, :-1, :].contiguous()

            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def calculate_loss(self, interaction):
        interaction = torch.tensor(interaction["input_ids"]).to(self.device)

        # BUG: the input_ids are not batched!!
        labels = interaction[:, 1:].contiguous()
        input_ids = interaction
        logits, loss = self.forward(input_ids, labels)

        return loss

    def predict(self, interaction):
        return self.forward(interaction["input_ids"])

    def full_sort_predict(self, interaction):
        predictions, probs = self.generate(interaction)
        scores, paths = self.get_sequences(interaction.size(0), predictions, probs)

        return scores, paths

    def _get_scores(self, sequences, scores):
        sequences_scores = None
        for i in range(scores.size(1)):
            tstep = scores[:, i]
            score = torch.softmax(tstep, dim=-1)
            if sequences_scores is None:
                sequences_scores = score[:, sequences[:, i]].sum(-1)
            else:
                sequences_scores += score[:, sequences[:, i]].sum(-1)

        return sequences_scores

    def get_sequences(self, batch_len, paths, probs):
        user_num = batch_len
        scores = torch.full((user_num, self.n_items), -torch.inf)
        user_topk_sequences = list()

        num_return_sequences = paths.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=paths.device).repeat_interleave(num_return_sequences)

        sequences_score = self._get_scores(paths[:, -self.ranker_max_new_tokens :], probs)

        for sequence, user_index, sequence_score in zip(paths, batch_user_index, sequences_score):
            seq = self.tokenizer.decode(sequence).split(" ")

            uid = int(seq[1][1:])
            recommended_token = seq[-1]

            if not recommended_token.startswith(PathLanguageModelingTokenType.ITEM.value):
                continue

            recommended_item = int(recommended_token[1:])

            if recommended_item in self.used_ids[uid]:
                continue

            scores[user_index, recommended_item] = max(scores[user_index, recommended_item], sequence_score)

            user_topk_sequences.append((uid, recommended_item, scores[user_index, recommended_item].item(), seq))

        return scores, user_topk_sequences

    @torch.no_grad()
    def generate(self, path, top_k=10):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # How many paths to return?
        path = path.repeat_interleave(self.path_gen_args["paths_per_user"], dim=0)
        scores = torch.full((path.size(0), self.ranker_max_new_tokens, len(self.tokenizer)), -float("Inf")).to(
            self.device
        )
        for i in range(self.ranker_max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.forward(path)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature

            # KGCD
            logits = self.logit_processor(path, logits)

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            scores[:, i] = probs
            # sample from the distribution
            path_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            path = torch.cat((path, path_next), dim=1)

        return path, scores
