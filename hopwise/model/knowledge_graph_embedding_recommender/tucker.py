# @Time   : 2024/11/21
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TuckER
##################################################
Reference:
    Balažević et al. "TuckER: Tensor Factorization for Knowledge Graph Completion." in EMNLP/IJCNLP 2019.

Reference code:
    https://github.com/ibalazevic/TuckER
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
        self.ui_relation = dataset.field2token_id["relation_id"][dataset.ui_relation]
        self.label_smoothing = config["label_smoothing"]
        self.input_dropout = config["input_dropout"]
        self.input_dropout1 = config["input_dropout1"]
        self.input_dropout2 = config["input_dropout2"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        self.weights = torch.nn.Parameter(
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

        # Loss
        self.loss = nn.BCELoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, h, r, embeddings):
        x = self.bn0(h)
        x = self.input_dropout(x)
        x = x.view(-1, 1, h.size(1))

        w_mat = torch.mm(r, self.weights.view(r.size(1), -1))
        w_mat = w_mat.view(-1, h.size(1), h.size(1))
        w_mat = self.hidden_dropout1(w_mat)

        x = torch.bmm(x, w_mat)
        x = x.view(-1, h.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, embeddings.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def _get_rec_embeddings(self, user):
        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        user_e = self.user_embedding(user)
        r_e = self.relation_embedding(relation_users)
        return user_e, r_e

    def _get_kg_embeddings(self, h, r):
        h = self.entity_embedding(h)
        r = self.relation_embedding(r)
        return h, r

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        item = interaction[self.ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        tail = interaction[self.TAIL_ENTITY_ID]

        head_e, relation_e = self._get_kg_embeddings(head, relation)
        user_e, rec_r_e = self._get_rec_embeddings(user)

        item_new = torch.zeros((item.size(0), self.n_users + self.n_items), device=self.device)
        item_new[:, item + self.n_users] = 1.0

        tail_new = torch.zeros((tail.size(0), self.n_entities), device=self.device)
        tail_new[:, tail] = 1.0

        if self.label_smoothing:
            item_new = ((1.0 - self.label_smoothing) * item_new) + (1.0 / self.n_items)
            tail_new = ((1.0 - self.label_smoothing) * tail_new) + (1.0 / self.n_entities)

        score_users = self.forward(user_e, rec_r_e, self.user_embedding)
        score_kg = self.forward(head_e, relation_e, self.entity_embedding)

        loss_rec = self.loss(score_users, item_new)
        loss_kg = self.loss(score_kg, tail_new)

        return loss_rec + loss_kg

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        user_e = self.user_embedding(user)
        r_e = self.relation_embedding(relation_users)

        score = self.forward(user_e, r_e, self.user_embedding)

        score = score[torch.arange(user.size(0)), item]
        return score

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        r_e = self.relation_embedding(relation)

        score = self.forward(head_e, r_e, self.entity_embedding)
        score = score[torch.arange(head.size(0)), tail]
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        user_e = self.user_embedding(user)
        r_e = self.relation_embedding(relation_users)

        score = self.forward(user_e, r_e, self.user_embedding)
        score = score[:, self.n_users :]
        return score

    # def full_sort_predict_kg(self, interaction):
    #     head = interaction[self.HEAD_ENTITY_ID]
    #     relation = interaction[self.RELATION_ID]

    #     head_e = self.entity_embedding(head)
    #     r_e = self.relation_embedding(relation)

    #     score = self.forward(head_e, r_e, self.entity_embedding)
    #     return score
