r"""KGREC
##################################################
Reference:
    Yuhao Yang et al. "Knowledge Graph Self-Supervised Rationalization for Recommendation" in WWW 2021.
Reference code:
    https://github.com/HKUDS/KGRec
"""


import math
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_mean, scatter_sum

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.model.layers import SparseDropout
from hopwise.model.loss import BPRLoss, EmbLoss
from hopwise.utils import InputType


class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7):
        super().__init__()
        self.tau: float = tau

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x):
            return torch.exp(x / self.tau)
        
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))

        return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        loss = self.loss(h1, h2).mean()
        return loss

class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """
    def __init__(
        self,
        embedding_size,
        n_hops,
        n_users,
        n_relations,
        edge_index,
        edge_type,
        inter_edge,
        inter_edge_w,
        node_dropout_rate=0.5,
        mess_dropout_rate=0.1
    ):
        super().__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()

        self.embedding_size = embedding_size
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.inter_edge=inter_edge,
        self.inter_edge_w = inter_edge_w,
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        # initializer = nn.init.xavier_uniform_
        # relation_emb = initializer(torch.empty(n_relations - 1, embedding_size))  # not include interact
        # self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.relation_emb = nn.Embedding(self.n_relations, self.embedding_size)


        self.W_Q = nn.Parameter(torch.Tensor(self.embedding_size, embedding_size))

        self.n_heads = 2
        self.d_k = self.embedding_size // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

    # def non_attn_agg(self, user_emb, entity_emb, relation_emb):

    #     n_entities = entity_emb.shape[0]

    #     """KG aggregate"""
    #     head, tail = self.edge_index
    #     edge_relation_emb = relation_emb[self.edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    #     neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
    #     entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

    #     """user aggregate"""
    #     item_agg = self.inter_edge_w.unsqueeze(-1) * entity_emb[self.inter_edge[1, :]]
    #     user_agg = scatter_sum(src=item_agg, index=self.inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
    #     return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg


    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, item_attn=None):

        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = self.inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge, 
                                                         inter_edge_w, self.relation_emb)

            """message dropout"""
            if self.mess_dropout:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.mess_dropout(item_emb)
                user_emb = self.mess_dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.mess_dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        if print:
            self.logger.info(f"edge_attn_score std: {edge_attn_score.std()}")
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score

class KGREC(KnowledgeRecommender):
    '''
        Losses:
        - bpr loss
        - masked autoencoder loss (mae)
        - contrastive loss
    '''
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.context_hops = config['context_hops']
        self.node_dropout_rate = ['node_dropout_rate']
        self.mess_dropout_rate = config['mess_dropout_rate']

        self.mae_coef = config['mae_coef']
        self.mae_msize = config['mae_msize']
        self.cl_coef = config['cl_coef']
        self.tau = config['cl_tau']
        self.cl_drop = config['cl_drop_ratio']

        self.samp_func = "torch"

        # hp_dict = None

        # if hp_dict is not None:
        #     self.load_other_parameter(hp_dict)

        self.interact_mat, _ = dataset._create_norm_ckg_adjacency_matrix(symmetric=False)
        self.inter_edge = self.interact_mat.indices().to(self.device)
        self.inter_edge_w = self.interact_mat.values().to(self.device)
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)

        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.gcn = AttnHGCN(embedding_size=self.embedding_size,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                       n_relations=self.n_relations,
                       edge_index=self.edge_index,
                       edge_type=self.edge_type,
                       inter_edge=self.inter_edge,
                       inter_edge_w = self.inter_edge_w,
                       node_dropout_rate=self.node_dropout_rate,
                       mess_dropout_rate=self.mess_dropout_rate)

        self.contrast_fn = Contrast(self.embedding_size, tau=self.tau)


        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def relation_aware_edge_sampling(self, edge_index, edge_type, n_relations, samp_rate=0.5):
        # exclude interaction
        for i in range(n_relations - 1):
            edge_index_i, edge_type_i = self.edge_sampling(
                edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
            if i == 0:
                edge_index_sampled = edge_index_i
                edge_type_sampled = edge_type_i
            else:
                edge_index_sampled = torch.cat(
                    [edge_index_sampled, edge_index_i], dim=1)
                edge_type_sampled = torch.cat(
                    [edge_type_sampled, edge_type_i], dim=0)
        return edge_index_sampled, edge_type_sampled
    
    def edge_sampling(edge_index, edge_type, samp_rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * samp_rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        topk_egde_id = topk_egde_id.cpu().numpy()
        topk_mask = np.zeros(n_edges, dtype=bool)
        topk_mask[topk_egde_id] = True
        # add another group of random mask
        random_indices = np.random.choice(
            n_edges, size=topk_egde_id.shape[0], replace=False)
        random_mask = np.zeros(n_edges, dtype=bool)
        random_mask[random_indices] = True
        # combine two masks
        mask = topk_mask | random_mask

        remain_edge_index = edge_index[:, ~mask]
        remain_edge_type = edge_type[~mask]
        masked_edge_index = edge_index[:, mask]
        masked_edge_type = edge_type[mask]

        return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

    def adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
        _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                        int((1-keep_rate) * edge_attn_score.shape[0]), sorted=False)
        cl_kg_mask = torch.ones_like(edge_attn_score).bool()
        cl_kg_mask[least_attn_edge_id] = False
        cl_kg_edge = edge_index[:, cl_kg_mask]
        cl_kg_type = edge_type[cl_kg_mask]
        return cl_kg_edge, cl_kg_type

    def adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func = "torch"):
        inter_attn_prob = item_attn_mean[inter_edge[1]]
        # add gumbel noise
        noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
        """ prob based drop """
        inter_attn_prob = inter_attn_prob + noise
        inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

        if samp_func == "np":
            # we observed abnormal behavior of torch.multinomial on mind
            sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]), size=int(keep_rate * inter_edge_w.shape[0]), replace=False, p=inter_attn_prob.cpu().numpy())
        else:
            sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

        return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx]/keep_rate

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        """node dropout"""
        # 1. graph sprasification;
        edge_index, edge_type = self.relation_aware_edge_sampling(self.edge_index, self.edge_type, self.n_relations,\
                                                                  self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, _ = self.gcn.norm_attn_computer(item_emb, edge_index, edge_type, return_logits=True)
        
        # for adaptive MAE training
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        _, topk_attn_edge_id = torch.topk(edge_attn_score, self.mae_msize, sorted=False)

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, _ = \
                        self.mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_attn_edge_id)

        # rec task
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb, item_emb, enc_edge_index, enc_edge_type, self.inter_edge,
                                                self.inter_edge_w, mess_dropout=self.mess_dropout_rate,)

        # MAE task with dot-product decoder
        # mask_size, 2, channel
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        # mask_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
        mae_loss = self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]

        # CL task
        """adaptive sampling"""
        cl_kg_edge, cl_kg_type = self.adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate=1-self.cl_drop)
        cl_ui_edge, cl_ui_w = self.adaptive_ui_drop_cl(item_attn_mean, self.inter_edge, self.inter_edge_w, \
                                                       1-self.cl_drop, samp_func=self.samp_func)
        item_agg_ui = self.gcn.forward_ui(user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        cl_loss = self.contrast_fn(item_agg_ui, item_agg_kg)

        # return user embeddings, entity/item embeddings, and edge-level rationale scores
        return user_gcn_emb, entity_gcn_emb, mae_loss, cl_loss
    
    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, mae_loss, cl_loss = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        # the three losses
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        bpr_loss = mf_loss + self.reg_weight * reg_loss
        mae_loss = self.mae_coef * mae_loss
        cl_loss = self.cl_coef * cl_loss

        total_loss = bpr_loss + mae_loss + cl_loss
        return total_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight                   # [n_items, batch_size]

        scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]

        return scores
    
