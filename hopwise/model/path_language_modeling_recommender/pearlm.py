# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

r"""PEARLM
##################################################
Reference:
    Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

Reference code:
    https://github.com/Chris1nexus/pearlm
"""

from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from hopwise.data import Interaction
from hopwise.model.abstract_recommender import ExplainablePathLanguageModelingRecommender
from hopwise.utils import PathLanguageModelingTokenType


class PEARLM(ExplainablePathLanguageModelingRecommender, GPT2LMHeadModel):
    """PEARLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
    as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
    representing a path. The model extends PLM by adding a constrained graph decoding mechanism to ensure that
    the generated paths are valid according to the knowledge graph structure. The model can be used for
    explainable recommendation by generating paths that explain the recommendations made by the model.
    """

    def __init__(self, config, dataset):
        ExplainablePathLanguageModelingRecommender.__init__(self, config, dataset)

        self.use_kg_token_types = config["use_kg_token_types"]

        transformers_config = AutoConfig.from_pretrained(
            "distilgpt2",
            **{
                "vocab_size": self.n_tokens,
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
        GPT2LMHeadModel.__init__(self, transformers_config)
        self.to(config["device"])

        # Add type ids template
        if self.use_kg_token_types:
            prev_vocab_size = self.n_tokens
            spec_type, spec_type_id = PathLanguageModelingTokenType.SPECIAL.value
            ent_type, ent_type_id = PathLanguageModelingTokenType.ENTITY.value
            rel_type, rel_type_id = PathLanguageModelingTokenType.RELATION.value

            token_types = [f"<Token-Type.{token_type}>" for token_type in [spec_type, ent_type, rel_type]]
            for token_type in token_types:
                dataset.tokenizer.add_tokens(token_type)
            self.n_tokens = len(dataset.tokenizer)  # Update the vocabulary size after adding new tokens

            spec_type_id, ent_type_id, rel_type_id = (
                spec_type_id + prev_vocab_size,
                ent_type_id + prev_vocab_size,
                rel_type_id + prev_vocab_size,
            )
            self.token_type_ids = torch.LongTensor(
                # BOS + ENT + REL + ENT + REL + ... + ENT + REL + EOS
                [spec_type_id, ent_type_id] + [rel_type_id, ent_type_id] * dataset.path_hop_length + [spec_type_id]
            )
            self.token_type_ids = self.token_type_ids.to(config["device"])

            self.transformer.resize_token_embeddings(self.n_tokens)

        self.loss = nn.CrossEntropyLoss()
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
        **kwargs,  # Additional arguments for compatibility with HuggingFace Trainer
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        if isinstance(input_ids, Interaction):
            token_type_ids = input_ids["token_type_ids"]
            attention_mask = input_ids["attention_mask"]
            input_ids = input_ids["input_ids"]

        if self.use_kg_token_types:
            # indexing is only relevant during generation to match token types length with input_ids
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

    def generate(self, inputs, **kwargs):
        kwargs["logits_processor"] = self.logits_processor_list
        kwargs["num_return_sequences"] = kwargs.pop("paths_per_user")
        return super(GPT2LMHeadModel, self).generate(**inputs, **kwargs)
