# @Time   : 2024/11/21
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TuckER
##################################################
Reference:
    TuckER: Tensor Factorization for Knowledge Graph Completion (Balazevic et al., EMNLP-IJCNLP 2019)
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class TuckER(KnowledgeRecommender):
    r"""TuckER relies on Tucker Decomposition. It handles entity and relation embeddings of independent dimension
    and jointly learns a share core W.

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.label_smoothing = config["label_smoothing"]
        self.input_dropout = config["input_dropout"]
        self.input_dropout1 = config["input_dropout1"]
        self.input_dropout2 = config["input_dropout2"]

        # items mapping
        self.items_indexes = torch.tensor(list(dataset.field2token_id["item_id"].values()), device=self.device)
        self.n_items = self.items_indexes.shape[0]

        # define layers and loss
        self.E_users = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.E_entities = nn.Embedding(self.n_entities, self.embedding_size)

        self.R = nn.Embedding(self.n_relations + 1, self.embedding_size)

        self.W = torch.nn.Parameter(
            torch.tensor(
                torch.empty(self.embedding_size, self.embedding_size, self.embedding_size).uniform_(-1, 1),
                requires_grad=True,
            )
        )

        self.input_dropout = torch.nn.Dropout(self.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(self.input_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(self.input_dropout2)

        self.bn0 = torch.nn.BatchNorm1d(self.embedding_size)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_size)

        self.rec_loss = nn.BCELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, relation, items):
        score = self._get_score_users(user, relation)
        score = score[:, self.n_users :]
        score = score[torch.arange(user.size(0)), items]
        return score

    def _get_score_users(self, h, r):
        e1 = self.E_users(h)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E_users.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def _get_score_entities(self, h, r):
        e1 = self.E_entities(h)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E_entities.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        item = interaction[self.ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        tail = interaction[self.TAIL_ENTITY_ID]

        relation_users = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        score_users = self._get_score_users(user, relation_users)

        score_heads = self._get_score_entities(head, relation)

        item_new = torch.zeros((item.size(0), self.n_users + self.n_items), device=self.device)
        item_new[:, item + self.n_users] = 1.0

        tail_new = torch.zeros((tail.size(0), self.n_entities), device=self.device)
        tail_new[:, tail] = 1.0

        if self.label_smoothing:
            item_new = ((1.0 - self.label_smoothing) * item_new) + (1.0 / self.items_indexes.shape[0])
            tail_new = ((1.0 - self.label_smoothing) * tail_new) + (1.0 / self.n_entities)

        loss_rec = self.rec_loss(score_users, item_new)
        loss_kg = self.rec_loss(score_heads, tail_new)

        return loss_rec + loss_kg

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        return self.forward(user, relation, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        relation = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        score = self._get_score_users(user, relation)
        score = score[:, self.n_users :]
        return score
