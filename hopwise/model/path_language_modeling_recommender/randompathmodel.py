from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from hopwise.data import Interaction
from hopwise.model.abstract_recommender import ExplainablePathLanguageModelingRecommender

class RandomPathModel(ExplainablePathLanguageModelingRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.to(config["device"])

    def forward(self, input_ids=None, return_dict=True, **kwargs):
        if isinstance(input_ids, Interaction):
            input_ids = input_ids["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.n_tokens, device=input_ids.device)
        if not return_dict:
            return (logits,)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def predict(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    @torch.no_grad()
    def generate(self, inputs, paths_per_user=1, max_length=None, **kwargs):
        input_ids = inputs["input_ids"]
        _, cur_len = input_ids.shape
        input_ids = input_ids.repeat_interleave(paths_per_user, dim=0)
        if max_length is None:
            max_length = cur_len
        extra_len = max_length - cur_len
        random_tokens = torch.randint(
            0,
            self.n_tokens,
            (input_ids.shape[0], extra_len),
            device=input_ids.device,
        )

        return torch.cat([input_ids, random_tokens], dim=1)