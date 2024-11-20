# @Time   : 2024/11/20
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""ComplEx
##################################################
Reference:
    Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48 (ICML'16). JMLR.org, 2071-2080.
"""

import torch
import torch.nn.functional as F
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

        # define layers and loss
        self.user_re_embedding = nn.Embedding(
            self.n_users, self.embedding_size)
        self.user_im_embedding = nn.Embedding(
            self.n_users, self.embedding_size)

        self.entity_re_embedding = nn.Embedding(
            self.n_entities, self.embedding_size)
        self.entity_im_embedding = nn.Embedding(
            self.n_entities, self.embedding_size)

        self.relation_re_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size)
        self.relation_im_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size)

        self.rec_loss = nn.BCEWithLogitsLoss()

        # items mapping
        self.items_indexes = torch.tensor(
            list(dataset.field2token_id['item_id'].values()), device=self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
    
    def forward(self, user, relation, item):
        score = self._get_score_users(user, relation, item)
        return score

    def _get_score_users(self, users, relations, items):
        user_re_e = self.user_re_embedding(users)
        item_re_e = self.entity_re_embedding(items)
        rec_r_re_e = self.relation_re_embedding(relations)

        user_im_e = self.user_im_embedding(users)
        item_im_e = self.entity_im_embedding(items)
        rec_r_im_e = self.relation_im_embedding(relations)

        return (self.triple_dot(user_re_e, rec_r_re_e, item_re_e)+self.triple_dot(user_im_e, rec_r_re_e, item_im_e)+self.triple_dot(user_re_e, rec_r_im_e, item_im_e)-self.triple_dot(user_im_e, rec_r_im_e, item_im_e))

    def triple_dot(self, x, y, z):
        return (x * y * z).sum(dim=-1)
    
    def _get_score_entities(self, heads, relations, tails):
        head_re_e = self.entity_re_embedding(heads)
        head_im_e = self.entity_im_embedding(heads)

        tail_re_e = self.entity_re_embedding(tails)
        tail_im_e = self.entity_im_embedding(tails)

        rec_r_re_e = self.relation_re_embedding(relations)
        rec_r_im_e = self.relation_im_embedding(relations)

        return (self.triple_dot(head_re_e, rec_r_re_e, tail_re_e)+self.triple_dot(head_im_e, rec_r_re_e, tail_im_e)+self.triple_dot(head_re_e, rec_r_im_e, tail_im_e)-self.triple_dot(head_im_e, rec_r_im_e, tail_im_e))

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        relation = interaction[self.RELATION_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        relations_users = torch.tensor(
            [self.n_relations]*user.shape[0], device=self.device)
        user_pos_interaction_score = self._get_score_users(
            user, relations_users, pos_item)
        user_neg_interaction_score = self._get_score_users(
            user, relations_users, neg_item)

        head_pos_interaction_score = self._get_score_entities(
            head, relation, pos_tail)
        head_neg_interaction_score = self._get_score_entities(
            head, relation, neg_tail)

        pos_scores = torch.cat(
            [user_pos_interaction_score, head_pos_interaction_score])
        neg_scores = torch.cat(
            [user_neg_interaction_score, head_neg_interaction_score])

        pos_labels = torch.ones_like(pos_scores).to(self.device)
        neg_labels = torch.zeros_like(neg_scores).to(self.device)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = self.rec_loss(scores, labels)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation = torch.tensor(
            [self.n_relations]*user.shape[0], device=self.device)
        return self.forward(user, relation, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        rec_r_re_e = self.relation_re_embedding.weight[-1]
        rec_r_im_e = self.relation_im_embedding.weight[-1]
        rec_r_re_e = rec_r_re_e.expand_as(user_re_e)
        rec_r_im_e = rec_r_im_e.expand_as(user_re_e)

        all_item_re_e = self.entity_re_embedding.weight[self.items_indexes]
        all_item_im_e = self.entity_im_embedding.weight[self.items_indexes]

        user_re_e = user_re_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)
        user_im_e = user_im_e.unsqueeze(1).expand(-1, all_item_re_e.shape[0], -1)

        rec_r_re_e = rec_r_re_e.unsqueeze(
            1).expand(-1, all_item_re_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(
            1).expand(-1, all_item_re_e.shape[0], -1)

        all_item_re_e = all_item_re_e.unsqueeze(0)
        all_item_im_e = all_item_im_e.unsqueeze(0)

        return (self.triple_dot(user_re_e, rec_r_re_e, all_item_re_e)+self.triple_dot(user_im_e, rec_r_re_e, all_item_im_e)+self.triple_dot(user_re_e, rec_r_im_e, all_item_im_e)-self.triple_dot(user_im_e, rec_r_im_e, all_item_im_e))
