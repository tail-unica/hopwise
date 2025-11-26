# @Time   : 2024/11/19
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""Analogy
##################################################
Reference:
    Liu et al. "Analogical Inference for Multi-Relational Embeddings." in ICML 2017.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.model.loss import LogisticLoss
from hopwise.utils import InputType


class Analogy(KnowledgeRecommender):
    r"""Analogy extends RESCAL so as to further model the analogical properties of entities and relations e.g.
    Interstellar is to Fantasy as Nolan is to Oppenheimer”.
    It employs the same scoring function as RESCAL but with some constraints.

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.scalar_share = config["scalar_share"]
        self.ui_relation = dataset.field2token_id["relation_id"][dataset.ui_relation]

        self.scalar_dim = int(self.embedding_size * self.scalar_share)
        self.complex_dim = int(self.embedding_size - self.scalar_dim)

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_re_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_im_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_re_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_im_embedding = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.relation_re_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.relation_im_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # Loss
        self.loss = LogisticLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, tail_e, tail_re_e, tail_im_e):
        return (head_e * r_e * tail_e).sum(dim=1) + (
            head_re_e * (r_re_e * tail_re_e + r_im_e * tail_im_e)
            + head_im_e * (r_re_e * tail_im_e - r_im_e * tail_re_e)
        ).sum(dim=1)

    def _get_rec_embeddings(self, users, pos_items, neg_items):
        user_e = self.user_embedding(users)
        user_re_e = self.user_re_embedding(users)
        user_im_e = self.user_im_embedding(users)

        pos_item_e = self.entity_embedding(pos_items)
        pos_item_re_e = self.entity_re_embedding(pos_items)
        pos_item_im_e = self.entity_im_embedding(pos_items)

        neg_item_e = self.entity_embedding(neg_items)
        neg_item_re_e = self.entity_re_embedding(neg_items)
        neg_item_im_e = self.entity_im_embedding(neg_items)

        relations = torch.tensor([self.ui_relation] * users.shape[0], device=self.device)
        rec_r_e = self.relation_embedding(relations)
        rec_r_re_e = self.relation_re_embedding(relations)
        rec_r_im_e = self.relation_im_embedding(relations)

        return (
            user_e,
            user_re_e,
            user_im_e,
            pos_item_e,
            pos_item_re_e,
            pos_item_im_e,
            neg_item_e,
            neg_item_re_e,
            neg_item_im_e,
            rec_r_e,
            rec_r_re_e,
            rec_r_im_e,
        )

    def _get_kg_embeddings(self, heads, relations, pos_tails, neg_tails):
        head_e = self.entity_embedding(heads)
        head_re_e = self.entity_re_embedding(heads)
        head_im_e = self.entity_im_embedding(heads)

        neg_tail_e = self.entity_embedding(neg_tails)
        neg_tail_re_e = self.entity_re_embedding(neg_tails)
        neg_tail_im_e = self.entity_im_embedding(neg_tails)

        pos_tail_e = self.entity_embedding(pos_tails)
        pos_tail_re_e = self.entity_re_embedding(pos_tails)
        pos_tail_im_e = self.entity_im_embedding(pos_tails)

        r_e = self.relation_embedding(relations)
        r_re_e = self.relation_re_embedding(relations)
        r_im_e = self.relation_im_embedding(relations)

        return (
            head_e,
            head_re_e,
            head_im_e,
            pos_tail_e,
            pos_tail_re_e,
            pos_tail_im_e,
            neg_tail_e,
            neg_tail_re_e,
            neg_tail_im_e,
            r_e,
            r_re_e,
            r_im_e,
        )

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        relation = interaction[self.RELATION_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        (
            user_e,
            user_re_e,
            user_im_e,
            pos_item_e,
            pos_item_re_e,
            pos_item_im_e,
            neg_item_e,
            neg_item_re_e,
            neg_item_im_e,
            rec_r_e,
            rec_r_re_e,
            rec_r_im_e,
        ) = self._get_rec_embeddings(user, pos_item, neg_item)
        (
            head_e,
            head_re_e,
            head_im_e,
            pos_tail_e,
            pos_tail_re_e,
            pos_tail_im_e,
            neg_tail_e,
            neg_tail_re_e,
            neg_tail_im_e,
            r_e,
            r_re_e,
            r_im_e,
        ) = self._get_kg_embeddings(head, relation, pos_tail, neg_tail)

        score_pos_users = self.forward(
            user_e, user_re_e, user_im_e, rec_r_e, rec_r_re_e, rec_r_im_e, pos_item_e, pos_item_re_e, pos_item_im_e
        )
        score_neg_users = self.forward(
            user_e, user_re_e, user_im_e, rec_r_e, rec_r_re_e, rec_r_im_e, neg_item_e, neg_item_re_e, neg_item_im_e
        )
        score_pos_kg = self.forward(
            head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, pos_tail_e, pos_tail_re_e, pos_tail_im_e
        )
        score_neg_kg = self.forward(
            head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, neg_tail_e, neg_tail_re_e, neg_tail_im_e
        )

        rec_loss = self.loss(-score_pos_users, -score_neg_users)
        kg_loss = self.loss(-score_pos_kg, -score_neg_kg)
        return rec_loss + kg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)

        user_e = self.user_embedding(user)
        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        r_e = self.relation_embedding(relation)
        r_re_e = self.relation_re_embedding(relation)
        r_im_e = self.relation_im_embedding(relation)

        item_e = self.entity_embedding(item)
        item_re_e = self.entity_re_embedding(item)
        item_im_e = self.entity_im_embedding(item)

        return self.forward(user_e, user_re_e, user_im_e, r_e, r_re_e, r_im_e, item_e, item_re_e, item_im_e)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        user_re_e = self.user_re_embedding(user)
        user_im_e = self.user_im_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_re_e = self.relation_re_embedding.weight[-1]
        rec_r_im_e = self.relation_im_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)
        rec_r_re_e = rec_r_re_e.expand_as(user_e)
        rec_r_im_e = rec_r_im_e.expand_as(user_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]
        all_item_re_e = self.entity_re_embedding.weight[item_indices]
        all_item_im_e = self.entity_im_embedding.weight[item_indices]

        user_e = user_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        user_re_e = user_re_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        user_im_e = user_im_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)

        rec_r_e = rec_r_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        rec_r_re_e = rec_r_re_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)

        all_item_e = all_item_e.unsqueeze(0)
        all_item_re_e = all_item_re_e.unsqueeze(0)
        all_item_im_e = all_item_im_e.unsqueeze(0)

        return (user_e * rec_r_e * all_item_e).sum(dim=-1) + (
            user_re_e * (rec_r_re_e * all_item_re_e + rec_r_im_e * all_item_im_e)
            + user_im_e * (rec_r_re_e * all_item_im_e - rec_r_im_e * all_item_re_e)
        ).sum(dim=-1)

    def predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        tail = interaction[self.TAIL_ENTITY_ID]

        head_e = self.entity_embedding(head)
        head_re_e = self.entity_re_embedding(head)
        head_im_e = self.entity_im_embedding(head)

        r_e = self.relation_embedding(relation)
        r_re_e = self.relation_re_embedding(relation)
        r_im_e = self.relation_im_embedding(relation)

        tail_e = self.entity_embedding(tail)
        tail_re_e = self.entity_re_embedding(tail)
        tail_im_e = self.entity_im_embedding(tail)

        return self.forward(head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, tail_e, tail_re_e, tail_im_e)

    def full_sort_predict_kg(self, interaction):
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        head_e = self.entity_embedding(head)
        head_re_e = self.entity_re_embedding(head)
        head_im_e = self.entity_im_embedding(head)

        rec_r_e = self.relation_embedding(relation)
        rec_r_re_e = self.relation_re_embedding(relation)
        rec_r_im_e = self.relation_im_embedding(relation)
        rec_r_e = rec_r_e.expand_as(head_e)
        rec_r_re_e = rec_r_re_e.expand_as(head_e)
        rec_r_im_e = rec_r_im_e.expand_as(head_e)

        entity_indices = torch.tensor(range(self.n_entities)).to(self.device)
        all_entities_e = self.entity_embedding.weight[entity_indices]
        all_entities_re_e = self.entity_re_embedding.weight[entity_indices]
        all_entities_im_e = self.entity_im_embedding.weight[entity_indices]

        head_e = head_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)
        head_re_e = head_re_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)
        head_im_e = head_im_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)

        rec_r_e = rec_r_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)
        rec_r_re_e = rec_r_re_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(1).expand(-1, all_entities_e.shape[0], -1)

        all_entities_e = all_entities_e.unsqueeze(0)
        all_entities_re_e = all_entities_re_e.unsqueeze(0)
        all_entities_im_e = all_entities_im_e.unsqueeze(0)

        return (head_e * rec_r_e * all_entities_e).sum(dim=-1) + (
            head_re_e * (rec_r_re_e * all_entities_re_e + rec_r_im_e * all_entities_im_e)
            + head_im_e * (rec_r_re_e * all_entities_im_e - rec_r_im_e * all_entities_re_e)
        ).sum(dim=-1)
