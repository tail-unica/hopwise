# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""RESCAL
##################################################
Reference:
    Nickel et al. "A three-way model for collective learning on multi-relational data." in ICML 2011.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class RESCAL(KnowledgeRecommender):
    r"""RESCAL associates each entity with a vector to capture its latent semantics.
    Each relation is represented as a matrix which models pairwise interactions between latent vectors

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
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size**2)

        # Loss
        self.loss = nn.MarginRankingLoss(margin=self.margin)

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head, relation, tail):
        hr = torch.matmul(head.view(-1, 1, 1, self.embedding_size), relation)
        hr = hr.view(-1, self.embedding_size)
        return (hr * tail).sum(dim=1)

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1].view(1, 1, self.embedding_size, self.embedding_size)
        rec_r_e = rec_r_e.repeat(user_e.shape[0], 1, 1, 1)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation).view(-1, 1, self.embedding_size, self.embedding_size)

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

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        pos_score = self.forward(h_e, r_e, pos_t_e)
        neg_score = self.forward(h_e, r_e, neg_t_e)

        loss = self.loss(pos_score, neg_score, torch.ones_like(pos_score).to(self.device))

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)
        rec_r_e = self.relation_embedding.weight[-1].view(1, 1, self.embedding_size, self.embedding_size)
        rec_r_e = rec_r_e.repeat(user_e.shape[0], 1, 1, 1)

        return self.forward(user_e, rec_r_e, item_e)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        tail_e = self.entity_embedding(tail)
        rec_r_e = self.relation_embedding(relation).view(
            relation.shape[0], 1, self.embedding_size, self.embedding_size
        )

        return self.forward(head_e, rec_r_e, tail_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1].view(1, 1, self.embedding_size, self.embedding_size)
        rec_r_e = rec_r_e.repeat(user_e.shape[0], 1, 1, 1)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        user_e = user_e.view(-1, 1, 1, self.embedding_size)
        hr = torch.matmul(user_e, rec_r_e)

        scores = torch.matmul(hr.squeeze(2), all_item_e.T)
        scores = scores.squeeze(1)
        return scores

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]

        head_e = self.entity_embedding(head)

        rec_r_e = self.relation_embedding(relation).view(
            relation.shape[0], 1, self.embedding_size, self.embedding_size
        )

        all_tail_e = self.entity_embedding.weight

        head_e = head_e.view(-1, 1, 1, self.embedding_size)
        hr = torch.matmul(head_e, rec_r_e)

        scores = torch.matmul(hr.squeeze(2), all_tail_e.T)
        scores = scores.squeeze(1)
        return scores
