from enum import IntEnum
from typing import Optional, Union

import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, GPT2LMHeadModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from hopwise.data import Interaction
from hopwise.model.abstract_recommender import ExplainableRecommender, PathLanguageModelingRecommender
from hopwise.trainer.hf_path_trainer import (
    ConstrainedLogitsProcessorWordLevel,
    CumulativeSequenceScoreRanker,
    get_tokenized_used_ids,
)
from hopwise.utils import PathLanguageModelingTokenType

TokenType = IntEnum("TokenType", [("SPECIAL", 0), ("ENTITY", 1), ("RELATION", 2)])


class PEARLM(PathLanguageModelingRecommender, GPT2LMHeadModel, ExplainableRecommender):
    """PEARLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
    from a knowledge graph as a next-token prediction task.
    """

    def __init__(self, config, dataset):
        tokenizer = dataset.tokenizer
        transformers_config = AutoConfig.from_pretrained(
            "distilgpt2",
            **{
                "vocab_size": len(tokenizer),
                "n_ctx": config["context_length"],
                "n_positions": config["context_length"],
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "n_embd": config["embedding_size"],
                "n_head": config["num_heads"],
                "n_layer": config["num_layers"],
            },
        )
        PathLanguageModelingRecommender.__init__(self, config, dataset)
        GPT2LMHeadModel.__init__(self, transformers_config)
        self.to(config["device"])

        self.max_hop_length = dataset.path_hop_length

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(
            num_embeddings=len(TokenType), embedding_dim=transformers_config.hidden_size
        )

        middle_token_types = [TokenType.ENTITY.value, TokenType.RELATION.value] * ((self.max_hop_length - 1) // 2)
        self.token_type_ids = torch.tensor(
            [[TokenType.SPECIAL.value] + middle_token_types + [TokenType.SPECIAL.value]],
            dtype=torch.long,
            device=self.device,
        )

        self.loss = nn.CrossEntropyLoss()
        self.post_init()

    def get_type_embeds(self, batch_size):
        self.token_type_ids = self.token_type_ids.to(self.type_embeddings.weight.device)
        type_ids = self.token_type_ids.expand(batch_size, -1)
        return self.type_embeddings(type_ids)

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

        batch_size = input_ids.shape[0]
        type_embeds = self.get_type_embeds(batch_size)

        if inputs_embeds is not None:
            inputs_embeds += type_embeds

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
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            lm_loss = self.loss(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + transformer_outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def predict(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    def explain(self, interaction):
        tokenized_used_ids = get_tokenized_used_ids(self.used_ids, self.tokenizer)
        token_sequence_length = (1 + self.model_config["path_hop_length"]) + self.model_config["path_hop_length"]
        paths_per_user = self.model_config["path_generation_args"]["paths_per_user"]

        ranker = CumulativeSequenceScoreRanker(
            self.tokenizer,
            self.used_ids,
            self.dataset.item_num,
            topk=10,
            max_new_tokens=token_sequence_length - 2,
        )
        logits_processor = LogitsProcessorList(
            [
                ConstrainedLogitsProcessorWordLevel(
                    self.dataset.get_tokenized_ckg(),
                    tokenized_used_ids,
                    token_sequence_length,
                    self.tokenizer,
                    paths_per_user,
                    task=ConstrainedLogitsProcessorWordLevel.RECOMMENDATION_TASK,
                )
            ]
        )
        self.model_config["path_generation_args"].pop("paths_per_user")
        outputs = self.generate(
            **interaction,
            max_length=token_sequence_length,
            min_length=token_sequence_length,
            num_return_sequences=paths_per_user,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
            **self.model_config["path_generation_args"],
        )
        scores, user_topk_sequences = ranker.get_sequences(interaction["input_ids"].size(0), outputs)
        paths = []
        for i, (user, sequence) in enumerate(user_topk_sequences.items()):
            product = int(sequence[-1][1:])
            paths.append([user, product, scores[i, product], sequence])
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
            for node_idx in range(0, len(path), 2):
                if path[node_idx].startswith(PathLanguageModelingTokenType.USER.value):
                    if not node_idx:
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
