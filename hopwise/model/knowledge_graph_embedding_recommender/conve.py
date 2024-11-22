# @Time   : 2024/11/22
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""ConvE
##################################################
Reference:
    Tim Dettmers and Pasquale Minervini and Pontus Stenetorp and Sebastian Riedel.
    Convolutional 2D Knowledge Graph Embeddings. CORR, 2017.
"""

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class ConvE(KnowledgeRecommender):
    r"""ConvE represent h,r,t in a subset of real number in d dimension. When scoring them,
    it concatenates and reshape h and r into a unique input [h;r]. This input is passed through
    a convolutional layers with a set of k filters and then through a dense layer with d neurons
    and a set of weight W. The output is finally combined with the tail embedding t
    using the dot product to produce the final score.

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
        self.hidden_dropout = config["hidden_dropout"]
        self.feature_dropout = config["feature_dropout"]
        self.embedding_dim1 = config["embedding_shape"]
        self.embedding_dim2 = self.embedding_size // self.embedding_dim1
        self.hidden_size = config["hidden_size"]
        self.use_bias = config["use_bias"]

        # items mapping
        self.items_indexes = torch.tensor(list(dataset.field2token_id["item_id"].values()), device=self.device)
        self.n_items = self.items_indexes.shape[0]

        # define layers and loss
        self.E_users = nn.Embedding(self.n_users + self.n_items, self.embedding_size, padding_idx=0)
        self.E_entities = nn.Embedding(self.n_entities, self.embedding_size, padding_idx=0)

        self.R = nn.Embedding(self.n_relations + 1, self.embedding_size, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(self.input_dropout)
        self.hidden_drop = torch.nn.Dropout(self.hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.feature_dropout)
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_size)
        self.register_parameter("b_users", nn.Parameter(torch.zeros(self.n_users + self.n_items)))
        self.register_parameter("b_entities", nn.Parameter(torch.zeros(self.n_entities)))
        self.fc = torch.nn.Linear(self.hidden_size, self.embedding_size)

        self.rec_loss = nn.BCELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, relation, items):
        score = self._get_score_users(user, relation)
        score = score[:, self.n_users :]
        score = score[torch.arange(user.size(0)), items]
        return score

    def _get_score_users(self, h, r):
        e1_embedded = self.E_users(h).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        rel_embedded = self.R(r).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E_users.weight.transpose(1, 0))
        x += self.b_users.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

    def _get_score_entities(self, h, r):
        e1_embedded = self.E_entities(h).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        rel_embedded = self.R(r).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E_entities.weight.transpose(1, 0))
        x += self.b_entities.expand_as(x)
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
