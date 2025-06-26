# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""ComplEx
##################################################
Reference:
    Trouillon et al. "Complex embeddings for simple link prediction." in ICML'16.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class ComplEx(KnowledgeRecommender):
    r"""ComplEx extends DistMult by introducing complex-valued embeddings.

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
        # define layers and loss
        self.user_re_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_im_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.entity_re_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_im_embedding = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_re_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.relation_im_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head_re_e, head_im_e, rec_r_re_e, rec_r_im_e, tail_re_e, tail_im_e):
        return (
            self.triple_dot(head_re_e, rec_r_re_e, tail_re_e)
            + self.triple_dot(head_im_e, rec_r_re_e, tail_im_e)
            + self.triple_dot(head_re_e, rec_r_im_e, tail_im_e)
            - self.triple_dot(head_im_e, rec_r_im_e, tail_im_e)
        )

    def triple_dot(self, x, y, z):
        return (x * y * z).sum(dim=-1)

    def _get_rec_embeddings(self, user, positive_items, negative_items):
        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        pos_item_re_e = self.entity_re_embedding(positive_items)
        pos_item_im_e = self.entity_im_embedding(positive_items)

        neg_item_re_e = self.entity_re_embedding(negative_items)
        neg_item_im_e = self.entity_im_embedding(negative_items)

        rec_r_re_e = self.relation_re_embedding(relation_users)
        rec_r_im_e = self.relation_im_embedding(relation_users)

        return user_re_e, user_im_e, rec_r_re_e, rec_r_im_e, pos_item_re_e, pos_item_im_e, neg_item_re_e, neg_item_im_e

    def _get_kg_embeddings(self, head, relation, positive_tails, negative_tails):
        head_re_e = self.entity_re_embedding(head)
        head_im_e = self.entity_im_embedding(head)

        pos_tail_re_e = self.entity_re_embedding(positive_tails)
        pos_tail_im_e = self.entity_im_embedding(positive_tails)

        neg_tail_re_e = self.entity_re_embedding(negative_tails)
        neg_tail_im_e = self.entity_im_embedding(negative_tails)

        kg_r_re_e = self.relation_re_embedding(relation)
        kg_r_im_e = self.relation_im_embedding(relation)

        return head_re_e, head_im_e, kg_r_re_e, kg_r_im_e, pos_tail_re_e, pos_tail_im_e, neg_tail_re_e, neg_tail_im_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        relation = interaction[self.RELATION_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_re_e, user_im_e, rec_r_re_e, rec_r_im_e, pos_item_re_e, pos_item_im_e, neg_item_re_e, neg_item_im_e = (
            self._get_rec_embeddings(user, pos_item, neg_item)
        )
        head_re_e, head_im_e, kg_r_re_e, kg_r_im_e, pos_tail_re_e, pos_tail_im_e, neg_tail_re_e, neg_tail_im_e = (
            self._get_kg_embeddings(head, relation, pos_tail, neg_tail)
        )

        score_pos_users = self.forward(user_re_e, user_im_e, rec_r_re_e, rec_r_im_e, pos_item_re_e, pos_item_im_e)
        score_neg_users = self.forward(user_re_e, user_im_e, rec_r_re_e, rec_r_im_e, neg_item_re_e, neg_item_im_e)
        score_pos_kg = self.forward(head_re_e, head_im_e, kg_r_re_e, kg_r_im_e, pos_tail_re_e, pos_tail_im_e)
        score_neg_kg = self.forward(head_re_e, head_im_e, kg_r_re_e, kg_r_im_e, neg_tail_re_e, neg_tail_im_e)

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
        relation = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        item_re_e = self.entity_re_embedding(item)
        item_im_e = self.entity_im_embedding(item)

        rec_r_re_e = self.relation_re_embedding(relation)
        rec_r_im_e = self.relation_im_embedding(relation)

        return self.forward(user_re_e, user_im_e, rec_r_re_e, rec_r_im_e, item_re_e, item_im_e)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_re_e = self.entity_re_embedding(head)
        head_im_e = self.entity_im_embedding(head)

        tail_re_e = self.entity_re_embedding(tail)
        tail_im_e = self.entity_im_embedding(tail)

        rec_r_re_e = self.relation_re_embedding(relation)
        rec_r_im_e = self.relation_im_embedding(relation)

        return self.forward(head_re_e, head_im_e, rec_r_re_e, rec_r_im_e, tail_re_e, tail_im_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        rec_r_re_e = self.relation_re_embedding.weight[-1]
        rec_r_im_e = self.relation_im_embedding.weight[-1]
        rec_r_re_e = rec_r_re_e.expand_as(user_re_e)
        rec_r_im_e = rec_r_im_e.expand_as(user_re_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_re_e = self.entity_re_embedding.weight[item_indices]
        all_item_im_e = self.entity_im_embedding.weight[item_indices]

        user_re_e = user_re_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)
        user_im_e = user_im_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)

        rec_r_re_e = rec_r_re_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)

        all_item_re_e = all_item_re_e.unsqueeze(0)
        all_item_im_e = all_item_im_e.unsqueeze(0)

        return (
            self.triple_dot(user_re_e, rec_r_re_e, all_item_re_e)
            + self.triple_dot(user_im_e, rec_r_re_e, all_item_im_e)
            + self.triple_dot(user_re_e, rec_r_im_e, all_item_im_e)
            - self.triple_dot(user_im_e, rec_r_im_e, all_item_im_e)
        )

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        head_re_e = self.entity_re_embedding(head)
        head_im_e = self.entity_im_embedding(head)

        rec_r_re_e = self.relation_re_embedding(relation)
        rec_r_im_e = self.relation_im_embedding(relation)

        entity_indices = torch.tensor(range(self.n_entities)).to(self.device)
        all_entity_re_e = self.entity_re_embedding.weight[entity_indices]
        all_entity_im_e = self.entity_im_embedding.weight[entity_indices]

        head_re_e = head_re_e.unsqueeze(1).expand(-1, all_entity_re_e.shape[0], -1)
        head_im_e = head_im_e.unsqueeze(1).expand(-1, all_entity_re_e.shape[0], -1)

        rec_r_re_e = rec_r_re_e.unsqueeze(1).expand(-1, all_entity_re_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(1).expand(-1, all_entity_re_e.shape[0], -1)

        all_entity_re_e = all_entity_re_e.unsqueeze(0)
        all_entity_im_e = all_entity_im_e.unsqueeze(0)

        return (
            self.triple_dot(head_re_e, rec_r_re_e, all_entity_re_e)
            + self.triple_dot(head_im_e, rec_r_re_e, all_entity_im_e)
            + self.triple_dot(head_re_e, rec_r_im_e, all_entity_im_e)
            - self.triple_dot(head_im_e, rec_r_im_e, all_entity_im_e)
        )
