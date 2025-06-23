# @Time   : 2025/5/29
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

r"""PLM
################################################
Reference:
     Shijie Geng et al. "Path Language Modeling over Knowledge Graphs for Explainable Recommendation." in WWW 2022.

Reference code:
    https://github.com/mirkomarras/kgglm
"""

from enum import IntEnum
from typing import Optional, Union

import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from hopwise.data import Interaction
from hopwise.model.abstract_recommender import PathLanguageModelingRecommender
from hopwise.utils import PathLanguageModelingTokenType

TokenType = IntEnum("TokenType", [("SPECIAL", 0), ("ENTITY", 1), ("RELATION", 2)])


class PLM(PathLanguageModelingRecommender, GPT2LMHeadModel):
    """PLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
    from a knowledge graph as a next-token prediction task and employs two feature transformations separately
    for entities and relations. Its decoding process is unbounded, meaning that it can generate paths that are
    not faithful to the knowledge graph structure, i.e., it can generate paths that do not exist in the KG.
    """

    def __init__(self, config, dataset):
        self.use_kg_token_types = config["use_kg_token_types"]
        transformers_config = AutoConfig.from_pretrained(
            "distilgpt2",
            **{
                "vocab_size": len(dataset.tokenizer),
                "n_ctx": config["context_length"],
                "n_positions": config["context_length"],
                "pad_token_id": dataset.tokenizer.pad_token_id,
                "bos_token_id": dataset.tokenizer.bos_token_id,
                "eos_token_id": dataset.tokenizer.eos_token_id,
                "n_embd": config["embedding_size"],
                "n_head": config["num_heads"],
                "n_layer": config["num_layers"],
            },
        )
        PathLanguageModelingRecommender.__init__(self, config, dataset)
        GPT2LMHeadModel.__init__(self, transformers_config)

        # Add type ids template
        prev_vocab_size = len(dataset.tokenizer)
        token_types = [f"<{token_type.name}>" for token_type in TokenType]
        for token_type in token_types:
            dataset.tokenizer.add_tokens(token_type)

        spec_type, ent_type, rel_type = TokenType.SPECIAL.value, TokenType.ENTITY.value, TokenType.RELATION.value
        spec_type, ent_type, rel_type = (
            spec_type + prev_vocab_size,
            ent_type + prev_vocab_size,
            rel_type + prev_vocab_size,
        )
        self.token_type_ids = torch.LongTensor(
            # BOS + ENT + REL + ENT + REL + ... + ENT + REL + EOS
            [spec_type, ent_type] + [rel_type, ent_type] * dataset.path_hop_length + [spec_type]
        )
        self.token_entity_type_id = ent_type
        self.token_relation_type_id = rel_type
        self.token_type_ids = self.token_type_ids.to(config["device"])

        self.transformer.resize_token_embeddings(len(dataset.tokenizer))

        vocab_inv = {v: k for k, v in dataset.tokenizer.get_vocab().items()}
        relation_mask = torch.tensor(
            [
                vocab_inv[i].startswith(PathLanguageModelingTokenType.RELATION.value)
                for i in range(self.config.vocab_size)
            ],
            dtype=torch.float32,
        )
        entity_mask = torch.tensor(
            [
                vocab_inv[i].startswith(PathLanguageModelingTokenType.USER.value)
                or vocab_inv[i].startswith(PathLanguageModelingTokenType.ITEM.value)
                or vocab_inv[i].startswith(PathLanguageModelingTokenType.ENTITY.value)
                for i in range(self.config.vocab_size)
            ],
            dtype=torch.float32,
        )

        self.entity_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.relation_head = torch.nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.entity_loss = nn.CrossEntropyLoss(weight=entity_mask)
        self.relation_loss = nn.CrossEntropyLoss(weight=relation_mask)

        self.to(config["device"])
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        if isinstance(input_ids, Interaction):
            token_type_ids = input_ids["token_type_ids"]
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]

        token_type_ids = self.token_type_ids[[*range(input_ids.shape[1] - 1), -1]]
        token_type_ids = token_type_ids.expand(input_ids.shape[0], -1)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache and labels is None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict or self.config.use_return_dict,
        )

        sequence_output = transformer_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            sequence_output = sequence_output.to(self.entity_head.weight.device)

        # Get logits from the two heads, first based on entity tokens, then on relation tokens
        lm_entity_scores = self.entity_head(sequence_output)
        lm_relation_scores = self.relation_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # entity head loss
            entity_scores_mask = (token_type_ids[0, :-1] != self.token_entity_type_id).to(lm_entity_scores.device)
            entity_labels_mask = (token_type_ids[0, 1:-1] != self.token_relation_type_id).to(lm_entity_scores.device)
            shifted_entity_scores = lm_entity_scores[:, :-1, :][:, entity_scores_mask, :].contiguous()
            entity_labels = labels[:, 1:-1][:, entity_labels_mask].contiguous()
            lm_loss = self.entity_loss(shifted_entity_scores.view(-1, self.config.vocab_size), entity_labels.view(-1))

            # relation head loss
            relations_scores_mask = (token_type_ids[0, 1:-1] != self.token_relation_type_id).to(
                lm_entity_scores.device
            )
            relation_labels_mask = (token_type_ids[0, 1:] != self.token_entity_type_id).to(lm_entity_scores.device)
            shifted_relation_scores = lm_relation_scores[:, 1:-1, :][:, relations_scores_mask, :].contiguous()
            relation_labels = labels[:, 1:][:, relation_labels_mask].contiguous()
            lm_loss += self.relation_loss(
                shifted_relation_scores.view(-1, self.config.vocab_size), relation_labels.view(-1)
            )

        lm_scores = lm_entity_scores
        relation_idx = torch.arange(1, input_ids.shape[1], 2, device=input_ids.device)
        lm_scores[:, relation_idx] = lm_relation_scores[:, relation_idx]

        if not return_dict:
            output = (lm_scores,) + transformer_outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def predict(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    def explain(self, interaction, ranker, logits_processor, **kwargs):
        # update paths per user from the newest config, otherwise, use the saved config
        paths_per_user = kwargs.get("paths_per_user", self.model_config["path_generation_args"]["paths_per_user"])
        token_sequence_length = kwargs.get(
            "token_sequence_length",
            (1 + self.model_config["path_hop_length"]) + self.model_config["path_hop_length"] + 1,
        )
        paths_gen_args = kwargs.get("path_gen_args", self.model_config["path_generation_args"])

        paths_gen_args.pop("paths_per_user")
        outputs = self.generate(
            **interaction,
            max_length=token_sequence_length,
            min_length=token_sequence_length,
            num_return_sequences=paths_per_user,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
            **paths_gen_args,
        )
        _, user_topk_sequences = ranker.get_sequences(interaction["input_ids"].size(0), outputs)
        paths = [[user, item, score, sequence] for user, item, score, sequence in user_topk_sequences]
        # # make explanations as pandas dataframe, then return the results
        df = pd.DataFrame(paths, columns=["user", "product", "score", "path"])
        return df

    def decode_path(self, paths):
        """Standardise path format"""
        new_paths = list()
        for user, product, score, path in paths:
            new_path = []
            # Process the path
            # U R I R I R I
            for node_idx in range(1, len(path) + 1, 2):
                if path[node_idx].startswith(PathLanguageModelingTokenType.USER.value):
                    if not node_idx - 1:
                        new_node = ("self_loop", "user", int(path[node_idx][1:]))
                    else:
                        new_node = (int(path[node_idx - 1][1:]), "user", int(path[node_idx][1:]))
                elif path[node_idx].startswith(PathLanguageModelingTokenType.ITEM.value):
                    new_node = (int(path[node_idx - 1][1:]), "item", int(path[node_idx][1:]))
                else:
                    # Is an entity
                    new_node = (int(path[node_idx - 1][1:]), "entity", int(path[node_idx][1:]))
                new_path.append(new_node)
            new_paths.append([user, product, score, new_path])
        return new_paths
