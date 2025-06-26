# @Time   : 2025/05/25
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

r"""PEARLMGPT2
##################################################
Reference:
    Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

Reference code:
    https://github.com/Chris1nexus/pearlm
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
"""

import math
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import PathLanguageModelingRecommender

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
        self.hidden_size = config["embedding_size"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        # Reduce the projection dim to match desired output dim
        self.head_dim = config["embedding_size"] // config["num_heads"]

        assert config["embedding_size"] % config["num_heads"] == 0

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
        self.c_fc = nn.Linear(config["embedding_size"], 4 * config["embedding_size"], bias=config["bias"])
        self.silu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["embedding_size"], config["embedding_size"], bias=config["bias"])
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
        self.layernorm_1 = LayerNorm(config["embedding_size"], bias=config["bias"])
        self.causal_attn = AutoregressiveSelfAttention(config)
        self.layernorm_2 = LayerNorm(config["embedding_size"], bias=config["bias"])
        self.feedforward = FeedForward(config)

    def forward(self, x):
        x = self.layernorm_1(x)
        x = x + self.causal_attn(x)
        x = self.layernorm_2(x)
        x = x + self.feedforward(x)

        return x


class PEARLMGPT2(PathLanguageModelingRecommender):
    """
    Low-level implementation of PEARLM model based on GPT-2 architecture that does not rely on HuggingFace tools.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset, _skip_nn_module_init=False)
        config["context_length"] = dataset.context_length

        self.config = config
        self.dataset = dataset
        self.tokenizer = dataset.tokenizer

        self.temperature = config["temperature"]
        self.path_gen_args = config["path_generation_args"]

        self.wte = nn.Embedding(len(self.tokenizer), config["embedding_size"])
        self.wpe = nn.Embedding(dataset.context_length, config["embedding_size"])
        self.wp_type_e = nn.Embedding(len(TokenType), config["embedding_size"])

        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_layers"])])
        self.layernorm = nn.LayerNorm(config["embedding_size"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

        self.lm_head = nn.Linear(config["embedding_size"], len(self.tokenizer), bias=False)

        # weight tying
        self.wte.weight = self.lm_head.weight

        self.loss = nn.CrossEntropyLoss()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["num_layers"]))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        bs, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=self.device)  # shape (t)

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
        x = tok_emb + pos_emb + type_emb
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        return x

    def calculate_loss(self, interaction):
        labels = interaction[:, 1:].contiguous()
        input_ids = interaction
        lm_output = self.forward(input_ids)

        logits = self.lm_head(lm_output)
        logits = logits[:, :-1, :].contiguous()

        return self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))

    def predict(self, interaction):
        interaction = interaction["input_ids"]
        lm_output = self.forward(interaction)
        logits = self.lm_head(lm_output[:, [-1], :])

        return logits

    @torch.no_grad()
    def generate(self, **kwargs):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        inputs = kwargs.get("inputs")
        topk = kwargs.get("top_k")
        logit_processor = kwargs.get("logit_processor")
        max_new_tokens = kwargs.get("max_new_tokens")
        paths_per_user = kwargs.get("paths_per_user")

        # How many paths to return?
        inputs["input_ids"] = inputs["input_ids"].repeat_interleave(paths_per_user, dim=0)
        scores = torch.full((inputs["input_ids"].size(0), max_new_tokens, len(self.tokenizer)), -torch.inf).to(
            self.device
        )
        for i in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.predict(inputs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature

            # KGCD
            logits = logit_processor(inputs["input_ids"], logits)

            # optionally crop the logits to only the top k options
            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            scores[:, i] = probs
            # sample from the distribution
            path_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            inputs["input_ids"] = torch.cat((inputs["input_ids"], path_next), dim=1)

        return inputs["input_ids"], torch.unbind(scores, dim=1)
