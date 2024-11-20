# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""RotatE
##################################################
Reference: Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang.
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
International Conference on Learning Representations, 2019.
"""

import math

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class RotatE(KnowledgeRecommender):
    r"""RotatE models relations as rotations in a complex latent space with h, r, t belonging
    to the set of d-dimensional complex numbers. The embedding for r belonging to the set of d-dimensional
    complex numbers, is a rotation vector: in all its elements, the phase conveys the rotation along that axis,
    and the modulus is equal to 1.

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
        self.user_embedding_im = nn.Embedding(self.n_users, self.embedding_size)

        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding_im = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)

        self.rec_loss = nn.BCEWithLogitsLoss()

        # items mapping
        self.items_indexes = torch.tensor(list(dataset.field2token_id["item_id"].values()), device=self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        nn.init.uniform_(self.relation_embedding.weight, 0, 2 * math.pi)

    def forward(self, user, relation, item):
        score = self._get_score_users(user, relation, item)
        return score

    def _get_score_users(self, user, relation, item):
        user_re = self.user_embedding(user)
        user_im = self.user_embedding_im(user)
        item_re = self.entity_embedding(item)
        item_im = self.entity_embedding_im(item)

        rel_theta = self.relation_embedding(relation)

        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * user_re - rel_im * user_im) - item_re
        im_score = (rel_re * user_im + rel_im * user_re) - item_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score

    def _get_score_entities(self, head, relation, tail):
        head_re = self.entity_embedding(head)
        head_im = self.entity_embedding_im(head)
        tail_re = self.entity_embedding(tail)
        tail_im = self.entity_embedding_im(tail)

        rel_theta = self.relation_embedding(relation)

        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        relation_user = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        pos_score_users = self._get_score_users(user, relation_user, pos_item)
        neg_score_users = self._get_score_users(user, relation_user, neg_item)

        pos_score_entities = self._get_score_entities(head, relation, pos_tail)
        neg_score_entities = self._get_score_entities(head, relation, neg_tail)

        pos_scores = torch.cat([pos_score_users, pos_score_entities])
        neg_scores = torch.cat([neg_score_users, neg_score_entities])

        pos_labels = torch.ones_like(pos_scores).to(self.device)
        neg_labels = torch.zeros_like(neg_scores).to(self.device)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = self.rec_loss(scores, labels)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation_user = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        return self.forward(user, relation_user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        relation_user = torch.tensor([self.n_relations] * user.shape[0], device=self.device)
        all_items = self.items_indexes

        user_re = self.user_embedding(user)
        user_im = self.user_embedding_im(user)

        item_re = self.entity_embedding(all_items)
        item_im = self.entity_embedding_im(all_items)

        rel_theta = self.relation_embedding(relation_user)

        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        user_re = user_re.unsqueeze(1).expand(-1, all_items.shape[0], -1)
        user_im = user_im.unsqueeze(1).expand(-1, all_items.shape[0], -1)

        rel_re = rel_re.unsqueeze(1).expand(-1, all_items.shape[0], -1)
        rel_im = rel_im.unsqueeze(1).expand(-1, all_items.shape[0], -1)

        item_re = item_re.unsqueeze(0)
        item_im = item_im.unsqueeze(0)

        re_score = (rel_re * user_re - rel_im * user_im) - item_re
        im_score = (rel_re * user_im + rel_im * user_re) - item_im
        complex_score = torch.stack([re_score, im_score], dim=3)
        score = torch.linalg.vector_norm(complex_score, dim=(2, 3))

        return self.margin - score
