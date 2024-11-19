# @Time   : 2024/11/19
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""Analogy
##################################################
Reference:
    Hanxiao Liu, Yuexin Wu, and Yiming Yang. Analogical Inference for Multi-Relational Embeddings. May 2017.
"""

import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.utils import InputType


class Analogy(KnowledgeRecommender):
    r"""Analogy extends RESCAL so as to further model the analogical properties of entities and relations e.g. Interstellar is to Fantasy as Nolan is to Oppenheimer”. It employs the same scoring function as RESCAL but with some constraints.

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]

        if config['scalar_share'] is None:
            self.scalar_share=0.5
        else:    
            self.scalar_share = config["scalar_share"] 
        
        self.scalar_dim = int(self.embedding_size * self.scalar_share)
        self.complex_dim = int((self.embedding_size - self.scalar_dim))

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_re_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_im_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_re_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_im_embedding = nn.Embedding(self.n_entities, self.embedding_size)

        self.relation_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)
        self.relation_re_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)
        self.relation_im_embedding = nn.Embedding(self.n_relations + 1, self.embedding_size)

        self.rec_loss = nn.Softplus()

        # items mapping
        self.items_indexes = torch.tensor(list(dataset.field2token_id['item_id'].values()), device=self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, relation, item):
        score = self._get_score_users(user,relation,item)
        return score

    def _get_score_users(self,users, relations, items):
        user_e = self.user_embedding(users)
        user_re_e=self.user_re_embedding(users)
        user_im_e = self.user_im_embedding(users)

        item_e = self.entity_embedding(items)
        item_re_e = self.entity_re_embedding(items)
        item_im_e = self.entity_im_embedding(items)
        
        rec_r_e = self.relation_embedding(relations)
        rec_r_re_e=self.relation_re_embedding(relations)
        rec_r_im_e=self.relation_im_embedding(relations)

        return ((user_e * rec_r_e * item_e).sum(dim=1) +
                (user_re_e * (rec_r_re_e * item_re_e + rec_r_im_e * item_im_e) + user_im_e * (rec_r_re_e * item_im_e - rec_r_im_e * item_re_e)).sum(dim=1))

    def _get_score_entities(self, heads, relations, tails):
        head_e = self.entity_embedding(heads)
        head_re_e = self.entity_re_embedding(heads)
        head_im_e = self.entity_im_embedding(heads)

        tail_e = self.entity_embedding(tails)
        tail_re_e = self.entity_re_embedding(tails)
        tail_im_e = self.entity_im_embedding(tails)

        rec_r_e = self.relation_embedding(relations)
        rec_r_re_e = self.relation_re_embedding(relations)
        rec_r_im_e = self.relation_im_embedding(relations)

        return ((head_e * rec_r_e * tail_e).sum(dim=1) +
                (head_re_e * (rec_r_re_e * tail_re_e + rec_r_im_e * tail_im_e) + head_im_e * (rec_r_re_e * tail_im_e - rec_r_im_e * tail_re_e)).sum(dim=1))

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        relation = interaction[self.RELATION_ID]
        
        head = interaction[self.HEAD_ENTITY_ID]
        
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        relations_users=torch.tensor([self.n_relations]*user.shape[0], device=self.device)
        user_pos_interaction_score=self._get_score_users(user, relations_users, pos_item)
        user_neg_interaction_score=self._get_score_users(user, relations_users, neg_item)

        head_pos_interaction_score=self._get_score_entities(head, relation, pos_tail)
        head_neg_interaction_score = self._get_score_entities(head, relation, neg_tail)

        pos_scores = torch.cat([user_pos_interaction_score, head_pos_interaction_score])
        neg_scores = torch.cat([user_neg_interaction_score, head_neg_interaction_score])
        
        pos_labels = torch.ones(pos_scores.shape).to(self.device)
        neg_labels = -torch.ones(neg_scores.shape).to(self.device)

        labels=torch.cat([pos_labels, neg_labels])
        scores=torch.cat([pos_scores, neg_scores])

        loss = torch.mean(self.rec_loss(-labels*scores))
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        relation = torch.tensor([self.n_relations]*user.shape[0],device=self.device)
        return self.forward(user, relation, item)

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

        all_item_e = self.entity_embedding.weight[self.items_indexes]
        all_item_re_e = self.entity_re_embedding.weight[self.items_indexes]
        all_item_im_e = self.entity_im_embedding.weight[self.items_indexes]

        user_e = user_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        user_re_e = user_re_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        user_im_e = user_im_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)

        rec_r_e = rec_r_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        rec_r_re_e = rec_r_re_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        rec_r_im_e = rec_r_im_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)

        all_item_e = all_item_e.unsqueeze(0)
        all_item_re_e = all_item_re_e.unsqueeze(0)
        all_item_im_e = all_item_im_e.unsqueeze(0)

        return ((user_e * rec_r_e * all_item_e).sum(dim=-1) + (user_re_e * (rec_r_re_e * all_item_re_e + rec_r_im_e * all_item_im_e) + user_im_e * (rec_r_re_e * all_item_im_e - rec_r_im_e * all_item_re_e)).sum(dim=-1))
