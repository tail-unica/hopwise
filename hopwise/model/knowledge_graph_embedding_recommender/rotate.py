# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""RotatE
##################################################
Reference:
    Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." in ICLR 2019.

Reference code:
    https://github.com/torchkge-team/torchkge
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

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.device = config["device"]
        self.ui_relation = dataset.field2token_id["relation_id"][dataset.ui_relation]

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_embedding_im = nn.Embedding(self.n_users, self.embedding_size)

        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding_im = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # Loss
        self.loss = nn.BCEWithLogitsLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        nn.init.uniform_(self.relation_embedding.weight, 0, 2 * math.pi)

    def forward(self, head_re, head_im, relation, tail_re, tail_im):
        rel_re, rel_im = torch.cos(relation), torch.sin(relation)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score

    def _get_rec_embeddings(self, user, positive_items, negative_items):
        user_re = self.user_embedding(user)
        user_im = self.user_embedding_im(user)
        pos_item_re = self.entity_embedding(positive_items)
        pos_item_im = self.entity_embedding_im(positive_items)

        neg_item_re = self.entity_embedding(negative_items)
        neg_item_im = self.entity_embedding_im(negative_items)

        relation_user = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        rec_r_e = self.relation_embedding(relation_user)

        return user_re, user_im, rec_r_e, pos_item_re, pos_item_im, neg_item_re, neg_item_im

    def _get_kg_embeddings(self, head, relation, positive_tails, negative_tails):
        head_re = self.entity_embedding(head)
        head_im = self.entity_embedding_im(head)
        pos_tail_re = self.entity_embedding(positive_tails)
        pos_tail_im = self.entity_embedding_im(positive_tails)

        neg_tail_re = self.entity_embedding(negative_tails)
        neg_tail_im = self.entity_embedding_im(negative_tails)

        kg_r_e = self.relation_embedding(relation)

        return head_re, head_im, kg_r_e, pos_tail_re, pos_tail_im, neg_tail_re, neg_tail_im

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_re, user_im, rec_r_e, pos_item_re, pos_item_im, neg_item_re, neg_item_im = self._get_rec_embeddings(
            user, pos_item, neg_item
        )
        head_re, head_im, kg_r_e, pos_tail_re, pos_tail_im, neg_tail_re, neg_tail_im = self._get_kg_embeddings(
            head, relation, pos_tail, neg_tail
        )

        score_pos_users = self.forward(user_re, user_im, rec_r_e, pos_item_re, pos_item_im)
        score_neg_users = self.forward(user_re, user_im, rec_r_e, neg_item_re, neg_item_im)
        score_pos_kg = self.forward(head_re, head_im, kg_r_e, pos_tail_re, pos_tail_im)
        score_neg_kg = self.forward(head_re, head_im, kg_r_e, neg_tail_re, neg_tail_im)

        scores_rec = torch.cat([score_pos_users, score_neg_users], dim=0)
        scores_kg = torch.cat([score_pos_kg, score_neg_kg], dim=0)
        labels_rec = torch.cat([torch.ones_like(score_pos_users), torch.zeros_like(score_neg_users)], dim=0)
        labels_kg = torch.cat([torch.ones_like(score_pos_kg), torch.zeros_like(score_neg_kg)], dim=0)

        rec_loss = self.loss(scores_rec, labels_rec)
        kg_loss = self.loss(scores_kg, labels_kg)

        return rec_loss + kg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation_user = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        user_re = self.user_embedding(user)
        user_im = self.user_embedding_im(user)
        item_re = self.entity_embedding(item)
        item_im = self.entity_embedding_im(item)

        rec_r_e = self.relation_embedding(relation_user)

        return self.forward(user_re, user_im, rec_r_e, item_re, item_im)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_re = self.entity_embedding(head)
        head_im = self.entity_embedding_im(head)
        tail_re = self.entity_embedding(tail)
        tail_im = self.entity_embedding_im(tail)

        rec_r_e = self.relation_embedding(relation)

        return self.forward(head_re, head_im, rec_r_e, tail_re, tail_im)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        relation_user = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        item_indices = torch.tensor(range(self.n_items)).to(self.device)

        user_re = self.user_embedding(user)
        user_im = self.user_embedding_im(user)

        item_re = self.entity_embedding(item_indices)
        item_im = self.entity_embedding_im(item_indices)

        rel_theta = self.relation_embedding(relation_user)

        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        user_re = user_re.unsqueeze(1).expand(-1, item_indices.shape[0], -1)
        user_im = user_im.unsqueeze(1).expand(-1, item_indices.shape[0], -1)

        rel_re = rel_re.unsqueeze(1).expand(-1, item_indices.shape[0], -1)
        rel_im = rel_im.unsqueeze(1).expand(-1, item_indices.shape[0], -1)

        item_re = item_re.unsqueeze(0)
        item_im = item_im.unsqueeze(0)

        re_score = (rel_re * user_re - rel_im * user_im) - item_re
        im_score = (rel_re * user_im + rel_im * user_re) - item_im
        complex_score = torch.stack([re_score, im_score], dim=3)
        score = torch.linalg.vector_norm(complex_score, dim=(2, 3))

        return self.margin - score

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]

        head_re = self.entity_embedding(head)
        head_im = self.entity_embedding_im(head)

        tail_re = self.entity_embedding.weight
        tail_im = self.entity_embedding_im.weight

        rel_theta = self.relation_embedding(relation)

        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        head_re = head_re.unsqueeze(1).expand(-1, self.entity_embedding.weight.shape[0], -1)
        head_im = head_im.unsqueeze(1).expand(-1, self.entity_embedding.weight.shape[0], -1)

        rel_re = rel_re.unsqueeze(1).expand(-1, self.entity_embedding.weight.shape[0], -1)
        rel_im = rel_im.unsqueeze(1).expand(-1, self.entity_embedding.weight.shape[0], -1)

        tail_re = tail_re.unsqueeze(0)
        tail_im = tail_im.unsqueeze(0)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=3)
        score = torch.linalg.vector_norm(complex_score, dim=(2, 3))

        return self.margin - score
