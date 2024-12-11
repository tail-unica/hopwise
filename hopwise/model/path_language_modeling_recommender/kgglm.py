from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# from hopwise.model.abstract_recommender import PathLanguageModelingRecommender

TokenType = IntEnum("TokenType", [("SPECIAL", 0), ("ENTITY", 1), ("RELATION", 2)])


# class KGGLM(PathLanguageModelingRecommender):
class KGGLM(GPT2LMHeadModel):
    """KGGLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
    from a knowledge graph as a next-token prediction task.
    """

    def __init__(self, config, dataset):
        super().__init__(config)
        self.max_hop_length = dataset.max_hop_length

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=len(TokenType), embedding_dim=config.hidden_size)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        middle_token_types = [TokenType.ENTITY.value, TokenType.RELATION.value] * ((self.max_hop_length - 1) // 2)
        self.token_type_ids = torch.tensor(
            [[TokenType.SPECIAL.value] + middle_token_types + [TokenType.SPECIAL.value]],
            dtype=torch.long,
            device=self.type_embeddings.weight.device,
        )

        self.loss = nn.CrossEntropyLoss()
        self.init_weights()

    def get_type_embeds(self, batch_size):
        type_ids = self.token_type_ids.expand(batch_size, -1)
        return self.type_embeddings(type_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
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
