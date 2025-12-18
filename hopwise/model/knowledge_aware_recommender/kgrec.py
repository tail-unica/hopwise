r"""KGREC
##################################################
Reference:
    Yuhao Yang et al. "Knowledge Graph Self-Supervised Rationalization for Recommendation" in WWW 2021.
Reference code:
    https://github.com/HKUDS/KGRec
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_uniform_initialization
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
        mess_dropout_rate=0.1,
    ):
        super().__init__()

        self.no_attn_convs = nn.ModuleList()

        self.embedding_size = embedding_size
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.mess_dropout_rate = mess_dropout_rate

        # interact relation is ignored
        self.relation_embedding = nn.Embedding(self.n_relations - 1, self.embedding_size)
        self.W_Q = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))

        self.n_heads = 2
        self.d_k = self.embedding_size // self.n_heads

        # nn.init.xavier_uniform_(self.W_Q)
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w):
        from torch_geometric.utils import softmax as scatter_softmax
        from torch_scatter import scatter_sum

        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * self.relation_embedding(edge_type).view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)

        neigh_relation_emb = entity_emb[tail] * self.relation_embedding(edge_type)  # [-1, embedding_size]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def forward(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, item_attn=None):
        from torch_geometric.utils import softmax as scatter_softmax
        from torch_scatter import scatter_sum

        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, embedding_size]
        user_res_emb = user_emb  # [n_users, embedding_size]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(
                user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w
            )

            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            user_res_emb = torch.add(user_res_emb, user_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)

        return user_res_emb, entity_res_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                item_emb = self.mess_dropout(item_emb)
                user_emb = self.mess_dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        from torch_scatter import scatter_sum

        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        from torch_scatter import scatter_mean

        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_embedding(edge_type)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, embedding_size]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, return_logits=False):
        from torch_geometric.utils import softmax as scatter_softmax
        from torch_scatter import scatter_sum

        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_embedding(edge_type).view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm

        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score


class KGRec(KnowledgeRecommender):
    r"""KGRec is a self-supervised knowledge-aware recommender that identifies and focuses on informative knowledge
    graph connections through an attentive rationalization mechanism. It combines generative masking reconstruction
    and contrastive learning tasks to highlight and align meaningful knowledge and interaction signals. By masking
    and rebuilding high-rationale edges while filtering noisy ones, KGRec learns more interpretable and noise-resistant
    recommendations.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.reg_weight = config["reg_weight"]
        self.context_hops = config["context_hops"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]

        self.mae_coef = config["mae_coef"]
        self.mae_msize = config["mae_msize"]
        self.cl_coef = config["cl_coef"]
        self.cl_tau = config["cl_tau"]
        self.cl_drop = config["cl_drop"]
        self.samp_func = config["samp_func"]

        self.inter_edge, _ = dataset._create_norm_ckg_adjacency_matrix(symmetric=False)
        self.inter_edge = self.inter_edge.to(self.device)
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")  # [n_entities, n_entities]
        # edge_index: [2, -1]; edge_type: [-1,]
        self.edge_index, self.edge_type = self.get_edges(self.kg_graph)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        self.gcn = AttnHGCN(
            embedding_size=self.embedding_size,
            n_hops=self.context_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            mess_dropout_rate=self.mess_dropout_rate,
        )

        self.contrast_fn = Contrast(self.embedding_size, tau=self.cl_tau)
        self.node_dropout = SparseDropout(p=self.node_dropout_rate)

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def forward(self):
        from torch_scatter import scatter_mean

        user_emb = self.user_embedding.weight
        entity_emb = self.entity_embedding.weight

        """node dropout"""
        # 1. graph sparsification;
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.relation_aware_edge_sampling(sampling_rate=self.node_dropout_rate)
            inter_edge = self.node_dropout(self.inter_edge)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            inter_edge = self.inter_edge
        inter_edge, inter_edge_w = inter_edge._indices(), inter_edge._values()

        # 2. compute rationale scores;
        edge_attn_score, _ = self.gcn.norm_attn_computer(entity_emb, edge_index, edge_type, return_logits=True)

        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.0] = 1.0
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.0] = 1.0
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[: self.n_items]

        # for adaptive MAE training
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        _, topk_attn_edge_id = torch.topk(edge_attn_score, self.mae_msize, sorted=False)

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, _ = self.mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id
        )

        # rec task
        user_gcn_emb, entity_gcn_emb = self.gcn(
            user_emb, entity_emb, enc_edge_index, enc_edge_type, inter_edge, inter_edge_w
        )

        # MAE task with dot-product decoder
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        masked_edge_emb = self.gcn.relation_embedding(masked_edge_type)
        mae_loss = self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # CL task
        """adaptive sampling"""
        cl_kg_edge, cl_kg_type = self.adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score)
        cl_ui_edge, cl_ui_w = self.adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w)
        item_agg_ui = self.gcn.forward_ui(user_emb, entity_emb[: self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(entity_emb, cl_kg_edge, cl_kg_type)[: self.n_items]
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
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

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
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings, require_pow=True)
        bpr_loss = mf_loss + self.reg_weight * reg_loss
        mae_loss = self.mae_coef * mae_loss
        cl_loss = self.cl_coef * cl_loss

        total_loss = bpr_loss + mae_loss + cl_loss
        return total_loss

    def relation_aware_edge_sampling(self, sampling_rate=0.5):
        # exclude interaction
        for i in range(self.n_relations - 1):
            edge_index_i, edge_type_i = self.edge_sampling(
                self.edge_index[:, self.edge_type == i],
                self.edge_type[self.edge_type == i],
                sampling_rate=sampling_rate,
            )
            if i == 0:
                edge_index_sampled = edge_index_i
                edge_type_sampled = edge_type_i
            else:
                edge_index_sampled = torch.cat([edge_index_sampled, edge_index_i], dim=1)
                edge_type_sampled = torch.cat([edge_type_sampled, edge_type_i], dim=0)
        return edge_index_sampled, edge_type_sampled

    def edge_sampling(self, edge_index, edge_type, sampling_rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * sampling_rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def mae_edge_mask_adapt_mixed(self, edge_index, edge_type, topk_egde_id):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        topk_egde_id = topk_egde_id.cpu().numpy()
        topk_mask = np.zeros(n_edges, dtype=bool)
        topk_mask[topk_egde_id] = True
        # add another group of random mask
        random_indices = np.random.choice(n_edges, size=topk_egde_id.shape[0], replace=False)
        random_mask = np.zeros(n_edges, dtype=bool)
        random_mask[random_indices] = True
        # combine two masks
        mask = topk_mask | random_mask

        remain_edge_index = edge_index[:, ~mask]
        remain_edge_type = edge_type[~mask]
        masked_edge_index = edge_index[:, mask]
        masked_edge_type = edge_type[mask]

        return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

    def adaptive_kg_drop_cl(self, edge_index, edge_type, edge_attn_score):
        keep_rate = 1 - self.cl_drop
        _, least_attn_edge_id = torch.topk(
            -edge_attn_score, int((1 - keep_rate) * edge_attn_score.shape[0]), sorted=False
        )
        cl_kg_mask = torch.ones_like(edge_attn_score).bool()
        cl_kg_mask[least_attn_edge_id] = False
        cl_kg_edge = edge_index[:, cl_kg_mask]
        cl_kg_type = edge_type[cl_kg_mask]
        return cl_kg_edge, cl_kg_type

    def adaptive_ui_drop_cl(self, item_attn_mean, inter_edge, inter_edge_w):
        keep_rate = 1 - self.cl_drop
        inter_attn_prob = item_attn_mean[inter_edge[1]]
        # add gumbel noise
        noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
        """ prob based drop """
        inter_attn_prob = inter_attn_prob + noise
        inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

        if self.samp_func == "np":
            # we observed abnormal behavior of torch.multinomial on mind
            sampled_edge_idx = np.random.choice(
                np.arange(inter_edge_w.shape[0]),
                size=int(keep_rate * inter_edge_w.shape[0]),
                replace=False,
                p=inter_attn_prob.cpu().numpy(),
            )
        else:
            sampled_edge_idx = torch.multinomial(
                inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False
            )

        return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx] / keep_rate

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = -torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.gcn(
            self.user_embedding.weight,
            self.entity_embedding.weight,
            self.edge_index,
            self.edge_type,
            self.inter_edge._indices(),
            self.inter_edge._values(),
        )

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.gcn(
                self.user_embedding.weight,
                self.entity_embedding.weight,
                self.edge_index,
                self.edge_type,
                self.inter_edge._indices(),
                self.inter_edge._values(),
            )

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
