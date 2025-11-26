# @Time   : 2024/11/12
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TransE
##################################################
Reference:
    Bordes. A et al. "Translating Embeddings for Modeling Multi-relational Data." in NeurIPS 2013.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class TransR(KnowledgeRecommender):
    r"""TransR Rather than introducing relation-specific hyperplanes, it introduces relation-specific spaces.
    The scoring functions is the same as TransH but h and t are projected into the space specific to relation

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
        self.proj_mat_e = nn.Embedding(self.n_relations, self.embedding_size * self.embedding_size)

        # Loss
        self.loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction="mean")

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)
        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, relation, pos_tail, neg_tail):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def forward(self, ent, proj_mat):
        proj_e = torch.matmul(proj_mat, ent.unsqueeze(2))
        proj_e = proj_e.squeeze(-1)
        return proj_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_rec_embedding(user, pos_item, neg_item)
        head_e, pos_tail_e, neg_tail_e, relation_e = self._get_kg_embedding(head, relation, pos_tail, neg_tail)

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        rec_rel = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        relation = torch.cat([rec_rel, relation])

        proj_mat = self.proj_mat_e(relation).view(h_e.shape[0], self.embedding_size, self.embedding_size)

        h_e_proj = self.forward(h_e, proj_mat)
        pos_t_e_proj = self.forward(pos_t_e, proj_mat)
        neg_t_e_proj = self.forward(neg_t_e, proj_mat)

        loss = self.loss(h_e_proj + r_e, pos_t_e_proj, neg_t_e_proj)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        relation = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        proj_mat = self.proj_mat_e(relation).view(user_e.shape[0], self.embedding_size, self.embedding_size)

        user_e_proj = self.forward(user_e, proj_mat)
        item_e_proj = self.forward(item_e, proj_mat)

        return -torch.norm(user_e_proj + rec_r_e - item_e_proj, p=2, dim=1)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        tail_e = self.entity_embedding(tail)

        rec_r_e = self.relation_embedding(relation)

        proj_mat = self.proj_mat_e(relation).view(head_e.shape[0], self.embedding_size, self.embedding_size)

        head_e_proj = self.forward(head_e, proj_mat)
        tail_e_proj = self.forward(tail_e, proj_mat)

        return -torch.norm(head_e_proj + rec_r_e - tail_e_proj, p=2, dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        relation_items = torch.tensor([self.ui_relation] * all_item_e.shape[0], device=self.device)

        proj_mat_user = self.proj_mat_e(relation_users).view(user.shape[0], self.embedding_size, self.embedding_size)
        proj_mat_items = self.proj_mat_e(relation_items).view(
            all_item_e.shape[0], self.embedding_size, self.embedding_size
        )

        user_e_proj = self.forward(user_e, proj_mat_user)
        item_e_proj = self.forward(all_item_e, proj_mat_items)

        user_e_proj = user_e_proj.unsqueeze(1).expand(-1, item_e_proj.shape[0], -1)
        rec_r_e = rec_r_e.unsqueeze(0).expand(1, item_e_proj.shape[0], -1)
        item_e_proj = item_e_proj.unsqueeze(0)

        return -torch.norm(user_e_proj + rec_r_e - item_e_proj, p=2, dim=2)
