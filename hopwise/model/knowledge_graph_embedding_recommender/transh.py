# @Time   : 2024/11/12
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TransH
##################################################
Reference:
    Wang Z. et al. Knowledge Graph Embedding by Translating on Hyperplanes.
    In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class TransH(KnowledgeRecommender):
    r"""TransH Have been invented to overcome the disadvantages of TransE,
    allowing an entity to have distinct representations when involved in different relations.
    It introduces relation-specific hyperplanes.

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
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.norm_vec = nn.Embedding(self.n_relations, self.embedding_size)

        # Loss
        self.rec_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction="mean")

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head, relation, tail, relation_ids):
        head_proj = self.project(head, relation_ids)
        tail_proj = self.project(tail, relation_ids)
        score = -torch.norm(head_proj + relation - tail_proj, p=2, dim=1)
        return score

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

    def project(self, ent, rel):
        return ent - (ent * self.norm_vec(rel).sum(1).view(-1, 1)) * self.norm_vec(rel)

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

        relation_user = torch.tensor([self.ui_relation] * user_e.shape[0], device=self.device)
        # Projections
        user_e = self.project(user_e, relation_user)
        head_e = self.project(head_e, relation)
        pos_item_e = self.project(pos_item_e, relation_user)
        pos_tail_e = self.project(pos_tail_e, relation)
        neg_item_e = self.project(neg_item_e, relation_user)
        neg_tail_e = self.project(neg_tail_e, relation)

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        loss = self.rec_loss(h_e + r_e, pos_t_e, neg_t_e)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)

        item_e = self.entity_embedding(item)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        relation_ids = torch.tensor([self.ui_relation] * user_e.shape[0], device=self.device)

        return self.forward(user_e, rec_r_e, item_e, relation_ids)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        relation_ids_user = torch.tensor([self.ui_relation] * user_e.shape[0], device=self.device)
        relation_ids_item = torch.tensor([self.ui_relation] * all_item_e.shape[0], device=self.device)

        h_r = self.project(user_e, relation_ids_user) + rec_r_e
        h_r = h_r.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)

        t = self.project(all_item_e, relation_ids_item)
        t = t.unsqueeze(0)
        return -torch.norm(h_r - t, p=2, dim=2)
