# @Time   : 2024/11/22
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""ConvE
##################################################
Reference:
    Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings." in AAAI 2018.

Reference code:
    https://github.com/TimDettmers/ConvE
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

        # Load parameters info
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
        self.ui_relation = dataset.field2token_id["relation_id"][dataset.ui_relation]

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size, padding_idx=0)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size, padding_idx=0)

        self.relations_embeddings = nn.Embedding(self.n_relations, self.embedding_size, padding_idx=0)

        # Layers
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

        # Loss
        self.loss = nn.BCELoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head, relation, embeddings, bias):
        stacked_inputs = torch.cat([head, relation], 2)
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
        x = torch.mm(x, embeddings.weight.transpose(1, 0))
        x += bias.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

    def _get_rec_embeddings(self, user):
        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        head_embeddings = self.user_embedding(user).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        relation_embeddings = self.relations_embeddings(relation_users).view(
            -1, 1, self.embedding_dim1, self.embedding_dim2
        )

        return head_embeddings, relation_embeddings

    def _get_kg_embeddings(self, head, relation):
        head_embeddings = self.entity_embedding(head).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        relation_embeddings = self.relations_embeddings(relation).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        return head_embeddings, relation_embeddings

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        item = interaction[self.ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        tail = interaction[self.TAIL_ENTITY_ID]

        user_e, rec_r_e = self._get_rec_embeddings(user)
        head_e, kg_r_e = self._get_kg_embeddings(head, relation)

        score_users = self.forward(user_e, rec_r_e, self.user_embedding, self.b_users)
        score_kg = self.forward(head_e, kg_r_e, self.entity_embedding, self.b_entities)

        items = torch.zeros((item.size(0), self.n_users + self.n_items), device=self.device)
        items[:, item + self.n_users] = 1.0

        tails = torch.zeros((tail.size(0), self.n_entities), device=self.device)
        tails[:, tail] = 1.0

        if self.label_smoothing:
            items = ((1.0 - self.label_smoothing) * items) + (1.0 / self.n_items)
            tails = ((1.0 - self.label_smoothing) * tails) + (1.0 / self.n_entities)

        rec_loss = self.loss(score_users, items)
        kg_loss = self.loss(score_kg, tails)

        return rec_loss + kg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        users_embedding = self.user_embedding(user).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        relation_embeddings = self.relations_embeddings(relation).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        score = self.forward(users_embedding, relation_embeddings, self.user_embedding, self.b_users)

        score = score[:, self.n_users :]
        score = score[torch.arange(user.size(0)), item]
        return score

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_embeddings = self.entity_embedding(head).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        relation_embeddings = self.relations_embeddings(relation).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        score = self.forward(head_embeddings, relation_embeddings, self.entity_embedding, self.b_entities)

        score = score[:, tail]
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        relation = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        users_embedding = self.user_embedding(user).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        relation_embeddings = self.relations_embeddings(relation).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        score = self.forward(users_embedding, relation_embeddings, self.user_embedding, self.b_users)
        score = score[:, self.n_users :]
        return score

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]

        head_embeddings = self.entities_embeddings(head).view(-1, 1, self.embedding_dim1, self.embedding_dim2)
        relation_embeddings = self.relations_embeddings(relation).view(-1, 1, self.embedding_dim1, self.embedding_dim2)

        score = self.forward(head_embeddings, relation_embeddings, self.entities_embeddings, self.b_entities)

        return score
