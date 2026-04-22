import torch
from torch import nn

from hopwise.data import Interaction
from hopwise.model.abstract_recommender import \
    ExplainablePathLanguageModelingRecommender
from hopwise.utils import GenerationOutputs


class RandomPathModel(ExplainablePathLanguageModelingRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset, _skip_nn_module_init=False)
        self.fake_parameters = torch.nn.Parameter(torch.zeros(1))
        self.loss = nn.CrossEntropyLoss()

    def calculate_loss(self, interaction):
        if isinstance(interaction, Interaction):
            input_ids = interaction["input_ids"]
        else:
            input_ids = interaction["input_ids"]

        labels = input_ids[:, 1:].contiguous()
        logits = self.forward(input_ids)
        logits = logits[:, :-1, :].contiguous()

        return self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))

    def forward(self, input_ids):
        if isinstance(input_ids, Interaction):
            input_ids = input_ids["input_ids"]
        elif isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.n_tokens, device=input_ids.device)
        return logits + self.fake_parameters

    def predict(self, inputs, **kwargs):
        return self.forward(inputs)

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
        sequences = torch.cat([input_ids, random_tokens], dim=1)
        scores = torch.rand((input_ids.shape[0], extra_len, self.n_tokens), device=input_ids.device)
        scores = torch.softmax(scores, dim=-1)

        return GenerationOutputs(sequences=sequences, scores=torch.unbind(scores, dim=1))