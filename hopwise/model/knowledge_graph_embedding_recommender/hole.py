# @Time   : 2024/11/21
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""HolE
##################################################
Reference:
    Nickel et al. "Holographic embeddings of knowledge graphs." in AAAI 2016.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class HolE(KnowledgeRecommender):
    r"""HoLE combines the expressive power of RESCAL with the efficiency and simplicity of DistMult.
    The entity representations are composed into h ⋆ t in the set of real numbers,
    with the circular correlation operator.

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.device = config["device"]

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # Loss
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MarginRankingLoss(margin=self.margin)

        # Embeddings Initialization
        self.apply(xavier_normal_initialization)

    def forward(self, h, r, t):
        r_e = self.get_rolling_matrix(r)
        hr = torch.matmul(h.view(-1, 1, self.embedding_size), r_e)
        return (hr.view(-1, self.embedding_size) * t).sum(dim=1)

    def get_rolling_matrix(self, x):
        b_size, dim = x.shape
        x = x.view(b_size, 1, dim)
        return torch.cat([x.roll(i, dims=2) for i in range(dim)], dim=1)

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_rec_embedding(user, pos_item, neg_item)
        head_e, pos_tail_e, neg_tail_e, relation_e = self._get_kg_embedding(head, pos_tail, neg_tail, relation)

        pos_score_users = self.forward(user_e, rec_r_e, pos_item_e)
        neg_score_users = self.forward(user_e, rec_r_e, neg_item_e)

        pos_score_entities = self.forward(head_e, relation_e, pos_tail_e)
        neg_score_entities = self.forward(head_e, relation_e, neg_tail_e)

        pos_scores = torch.cat([pos_score_users, pos_score_entities])
        neg_scores = torch.cat([neg_score_users, neg_score_entities])

        loss = self.loss(self.sigmoid(pos_scores), self.sigmoid(neg_scores), torch.ones_like(pos_scores))
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return self.forward(user_e, rec_r_e, item_e)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        tail_e = self.entity_embedding(tail)
        rec_r_e = self.relation_embedding(relation)

        return self.forward(head_e, rec_r_e, tail_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        r_e = self.get_rolling_matrix(rec_r_e)

        h_e = user_e.view(user_e.shape[0], 1, self.embedding_size)
        hr = torch.matmul(h_e, r_e).view(user_e.shape[0], self.embedding_size, 1)

        return torch.matmul(hr.squeeze(2), all_item_e.T)

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]

        head_e = self.entity_embedding(head)
        rec_r_e = self.relation_embedding(relation)

        all_item_e = self.entity_embedding.weight

        r_e = self.get_rolling_matrix(rec_r_e)

        h_e = head_e.view(head_e.shape[0], 1, self.embedding_size)
        hr = torch.matmul(h_e, r_e).view(head_e.shape[0], self.embedding_size, 1)

        return torch.matmul(hr.squeeze(2), all_item_e.T)
