# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""DistMult
##################################################
Reference: Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng: Embedding Entities and Relations for Learning
and Inference in Knowledge Bases. ICLR (Poster) 2015

"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class DistMult(KnowledgeRecommender):
    r"""DistMult simplify RESCAL by restricting Mr to diagonal matrices.
    For each relation r, it introduce a vector embedding r and requires Mr = diag(r).

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.device = config["device"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)

        self.loss = nn.MarginRankingLoss(margin=self.margin)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head, relation, tail):
        return (head * relation * tail).sum(dim=1)

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
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return self.forward(user_e, rec_r_e, item_e)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        item_e = self.entity_embedding(tail)
        rec_r_e = self.relation_embedding(relation)

        return self.forward(head_e, rec_r_e, item_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        h = user_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        r = rec_r_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        t = all_item_e.unsqueeze(0)

        return (h * r * t).sum(dim=-1)

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]

        head_e = self.entity_embedding(head)
        rec_r_e = self.relation_embedding(relation)

        h = head_e.unsqueeze(1).expand(-1, self.entity_embedding.weight.size(0), -1)
        r = rec_r_e.unsqueeze(1).expand(-1, self.entity_embedding.weight.size(0), -1)
        t = self.entity_embedding.weight.unsqueeze(0)

        return (h * r * t).sum(dim=-1)
