# @Time   : 2024/11/14
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TransD
##################################################
Reference:
    Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. Knowledge Graph Embedding via Dynamic Mapping Matrix. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) pages 687-696, Beijing, China, July 2015. Association for Computational Linguistics.
"""

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class TransD(KnowledgeRecommender):
    r"""TransD simplifies TransR by further decomposing the projection matrix into a product of two vector. Also in this case, the scoring function is the same as TransH and TransR, but it introduces three additional mapping vectors along with the entity and relation representation.
    
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
        
        self.user_vec_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_vec_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_vec_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)
        
        self.rec_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction="mean")

        # items mapping
        self.items_indexes = torch.tensor(list(dataset.field2token_id['item_id'].values()), device=self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        user_e_vec = self.user_vec_embedding(user)

        item_e = self.entity_embedding(item)
        item_e_vec = self.entity_vec_embedding(item)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        rec_r_e_vec = self.relation_vec_embedding.weight[-1]

        user_projection = self.project(user_e, user_e_vec, rec_r_e_vec)
        item_projection = self.project(item_e, item_e_vec, rec_r_e_vec)

        score = self._get_score(user_projection, rec_r_e, item_projection)
        return score

    def _get_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_rec_vec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_vec_embedding(user)
        pos_item_e = self.entity_vec_embedding(pos_item)
        neg_item_e = self.entity_vec_embedding(neg_item)
        rec_r_e = self.relation_vec_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def _get_kg_vec_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_vec_embedding(head)
        pos_tail_e = self.entity_vec_embedding(pos_tail)
        neg_tail_e = self.entity_vec_embedding(neg_tail)
        relation_e = self.relation_vec_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e
    
    def project(self, ent, ent_vect, rel_vect):
        """We note that :math:`p_r(e)_i = e^p^Te \\times r^p_i + e_i` which is
        more efficient to compute than the matrix formulation in the original
        paper. """
        proj_e = (rel_vect * (ent * ent_vect).sum(dim=1).view(ent.shape[0], 1))
        return proj_e + ent[:, :self.embedding_size]

    def _get_score(self, h_e, r_e, t_e):
        return -torch.norm(h_e + r_e - t_e, p=2, dim=1)

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

        user_e_vec, pos_item_e_vec, neg_item_e_vec, rec_r_e_vec = self._get_rec_vec_embedding(user, pos_item, neg_item)
        head_e_vec, pos_tail_e_vec, neg_tail_e_vec, relation_e_vec = self._get_kg_vec_embedding(head, pos_tail, neg_tail, relation)

        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        h_e_vec = torch.cat([user_e_vec, head_e_vec])
        r_e_vec = torch.cat([rec_r_e_vec, relation_e_vec])
        pos_t_e_vec = torch.cat([pos_item_e_vec, pos_tail_e_vec])
        neg_t_e_vec = torch.cat([neg_item_e_vec, neg_tail_e_vec])

        h_projection = self.project(h_e, h_e_vec, r_e_vec)
        pos_t_e_projection = self.project(pos_t_e, pos_t_e_vec, r_e_vec)
        neg_t_e_projection = self.project(neg_t_e, neg_t_e_vec, r_e_vec)

        loss = self.rec_loss(h_projection + r_e, pos_t_e_projection, neg_t_e_projection)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        user_e_vec = self.user_vec_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]

        rec_r_e_vec = self.relation_vec_embedding.weight[-1]

        users_projection = self.project(user_e, user_e_vec, rec_r_e_vec)

        all_item_e = self.entity_embedding.weight[self.items_indexes]
        all_item_e_vec = self.entity_vec_embedding.weight[self.items_indexes]
        items_projection = self.project(all_item_e, all_item_e_vec, rec_r_e_vec)

        h_r = (users_projection+rec_r_e).unsqueeze(1).expand(-1, items_projection.shape[0], -1)
        t = items_projection.unsqueeze(0)

        return -torch.norm(h_r-t, p=2, dim=2)
