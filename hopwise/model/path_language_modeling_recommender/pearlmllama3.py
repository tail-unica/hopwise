# @Time   : 2025/05
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

r"""PEARLMLlama3
##################################################
Reference:
    Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

Reference code:
    https://github.com/Chris1nexus/pearlm
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb
"""

import math
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import PathLanguageModelingRecommender

TokenType = IntEnum("TokenType", [("SPECIAL", 0), ("USER", 1), ("ENTITY", 2), ("RELATION", 3)])


class AutoregressiveGroupQuerySelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["embedding_size"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]

        # Reduce the projection dim to match desired output dim
        self.head_dim = config["embedding_size"] // config["num_heads"]

        assert config["embedding_size"] % config["num_heads"] == 0

        # the second hidden size could be different
        self.W_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=config["weight_precision"])
        self.W_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=config["weight_precision"])

        num_kv_groups = config["num_kv_groups"]
        self.group_size = self.num_heads // num_kv_groups

        self.W_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=config["weight_precision"])
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=config["weight_precision"])

        # RoPE pe
        self.mask, self.cos, self.sin = SharedBuffers.get_buffers(
            config["context_length"],
            self.head_dim,
            config["rope_base"],
            config["rope_config"],
            config["weight_precision"],
        )

        self.mask, self.cos, self.sin = (
            self.mask.to(config["device"]),
            self.cos.to(config["device"]),
            self.sin.to(config["device"]),
        )

        self.causal_mask = (
            torch.triu(torch.ones(config["context_length"], config["context_length"]), diagonal=1)
            .bool()
            .to(config["device"])
        )

    def forward(self, x):
        batch_size, seq_length, hidden_size = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward
        # to be the batch dim

        # applies a linear transformation on the last dimension: xW^T = (9,256)(256,256)
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        k = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        q = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # NOTE: Fetch ROPE PE Embeddings
        k = compute_rope(k, self.cos, self.sin)
        q = compute_rope(q, self.cos, self.sin)

        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Dot product for each head
        # Calculate attention scores

        attn_scores = (q @ k.transpose(2, 3)) / (1.0 / math.sqrt(k.size(-1)))

        # apply causal masking
        causal_mask = self.causal_mask.bool()[:seq_length, :seq_length]
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)

        context_vec = (attn_scores @ v).transpose(1, 2)

        context_vec = context_vec.reshape(batch_size, seq_length, self.hidden_size)
        context_vec = self.out_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["embedding_size"], config["embedding_size"], bias=False)
        self.fc2 = nn.Linear(config["embedding_size"], config["embedding_size"], bias=False)
        self.fc3 = nn.Linear(config["embedding_size"], config["embedding_size"], bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rmsnorm1 = nn.RMSNorm(config["embedding_size"], eps=1e-5)
        self.causal_attn = AutoregressiveGroupQuerySelfAttention(config)
        self.rmsnorm2 = nn.RMSNorm(config["embedding_size"], eps=1e-5)
        self.feedforward = FeedForward(config)

    def forward(self, x):
        # NOTE: Attention Block
        shortcut = x
        x = self.rmsnorm1(x)
        x = self.causal_attn(x)
        x += shortcut

        # NOTE: Feed Forward Block
        shortcut = x
        x = self.rmsnorm2(x)
        x = self.feedforward(x)
        x = x + shortcut
        return x


class PEARLMLlama3(PathLanguageModelingRecommender):
    """
    Low-level implementation of PEARLM model based on LLaMA 3 architecture.

    With 8 kv-groups (that's how many Llama 3 8B uses), we can see that the number of rows
    of the key and value matrices are reduced by a factor of 4
    (because 32 attention heads divided by 8 kv-groups is 4)
    To make the GroupedQueryAttention equivalent to standard multi-head attention,
    you can set the number of query groups equal to the number of heads.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset, _skip_nn_module_init=False)
        config["context_length"] = dataset.context_length

        self.config = config
        self.dataset = dataset
        self.tokenizer = dataset.tokenizer

        self.temperature = config["temperature"]

        self.wte = nn.Embedding(len(self.tokenizer), config["embedding_size"]).to(dtype=config["weight_precision"])
        self.wpe = nn.Embedding(len(TokenType), config["embedding_size"]).to(dtype=config["weight_precision"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_layers"])])
        self.rmsnorm = nn.RMSNorm(config["embedding_size"], eps=1e-5)

        self.lm_head = nn.Linear(config["embedding_size"], len(self.tokenizer), bias=False).to(
            dtype=config["weight_precision"]
        )

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

    def forward(self, idx, targets=None):
        device = idx.device
        bs, t = idx.size()
        # noqa: PLR2004 assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.wte(idx)
        # position embeddings of shape (t, n_embd)
        pos = torch.tensor(
            # BOS + USER + REL + ENT + REL + ... + ENT + REL + EOS
            [TokenType.SPECIAL.value, TokenType.USER.value]
            + [TokenType.RELATION.value, TokenType.ENTITY.value] * self.config["path_hop_length"]
            + [TokenType.SPECIAL.value]
        ).to(self.device)

        # think about get_flops, it restrict the max length of the input
        pos_emb = self.wpe(pos)[: tok_emb.size(1)]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x.to(self.config["weight_precision"]))
        x = self.rmsnorm(x)

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


# RoPE


def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None, device=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments used in LLaMA 3.1 and 3.2
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq)

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    # Shape: (context_length, head_dim // 2)
    angles = positions[:, None] * inv_freq[None, :]

    # Expand angles to match the head_dim
    # Shape: (context_length, head_dim)
    angles = torch.cat([angles, angles], dim=1)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.to(device), sin.to(device)


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated


class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]
