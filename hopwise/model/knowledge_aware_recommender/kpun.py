# @Time   : 2025/09/29
# @Author : Gador Mostallino
# @Email  : g.mostallino@studenti.unica.it

r"""KPUN
##################################################

Reference:
    Xiang Wang et al. "Learning Intents behind Interactions with Knowledge Graph for Recommendation." in WWW 2021.
    Caijun Xu et el. Exploring High-Order User Preference with Knowledge Graph for Recommendation. in 2024

Reference code:
    https://github.com/molumolua/KUPN
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_uniform_initialization
from hopwise.model.layers import SparseDropout
from hopwise.model.loss import BPRLoss
from hopwise.utils import InputType


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        all_emb,
        edge_index,
        edge_type,
        weight,
        relation_emb,
        aug_edge_weight=None,
        div=False,
        with_relation=True
    ):
        from torch_scatter import scatter_mean, scatter_sum
    
        """aggregate"""
        dim=all_emb.shape[0]
        head, tail = edge_index 
        if with_relation:
            edge_relation_emb = relation_emb[edge_type] 
            neigh_relation_emb = all_emb[tail] * edge_relation_emb 
        else:
            neigh_relation_emb = all_emb[tail]
            
        if aug_edge_weight is not None:
            neigh_relation_emb *= aug_edge_weight
        if div:
            res_emb = scatter_mean(neigh_relation_emb, head, dim_size=dim, dim=0)
        else:
            res_emb = scatter_sum(neigh_relation_emb, head, dim_size=dim, dim=0)
        return res_emb
    
    def batch_get_contribute(
        self,
        all_emb,
        edge_index,
        edge_type,
        weight,
        mask,
        batch_size,
        aug_edge_weight,
        rate, 
        with_relation=True
    ):
        dim = all_emb.shape[0]
        channel = all_emb.shape[1]
        head, tail = edge_index
        head=head[mask]
        tail=tail[mask]
        edge_type=edge_type[mask]
        aug_edge_weight=aug_edge_weight[mask]
        n_batches = (edge_index.shape[1] + batch_size - 1) // batch_size
        contrib_sum = torch.zeros(dim, channel).to(edge_index.device)
        degrees = torch.zeros(dim).to(edge_index.device)
        for b in range(n_batches):
            start_idx = b * batch_size
            # print(start_idx,edge_index.shape[1])
            end_idx = min((b + 1) * batch_size, edge_index.shape[1])

            head_batch = head[start_idx:end_idx]
            tail_batch = tail[start_idx:end_idx]

            if with_relation:
                edge_type_batch = edge_type[start_idx:end_idx]
                edge_relation_emb_batch = weight[edge_type_batch]
                neigh_relation_emb_batch = all_emb[tail_batch] * edge_relation_emb_batch
            else:
                neigh_relation_emb_batch = all_emb[tail_batch]

            if aug_edge_weight is not None:
                aug_edge_weight_batch = aug_edge_weight[start_idx:end_idx]
                neigh_relation_emb_batch *= aug_edge_weight_batch.unsqueeze(-1)


            contrib_sum.index_add_(0, head_batch, neigh_relation_emb_batch*rate)

            degrees.index_add_(0, head_batch, torch.ones_like(head_batch, dtype=torch.float)*rate)
        return degrees,contrib_sum
        
        
    def batch_generate(
        self, 
        all_emb,
        edge_index,
        edge_type,
        weight,
        aug_edge_weight=None,
        batch_size=16777216,
        zero_rate=1.0,
        div=False,
        with_relation=True
    ):
        """aggregate"""

        zero_mask=(edge_type == 0) | (edge_type == self.n_prefers/2)
        zero_degrees,zero_contrib_sum=self.batch_get_contribute(all_emb,edge_index,edge_type,weight,zero_mask,batch_size,aug_edge_weight,zero_rate,with_relation)


        nonzero_mask=~ zero_mask # user entity
        nonzero_degrees,nonzero_contrib_sum=self.batch_get_contribute(all_emb,edge_index,edge_type,weight,nonzero_mask,batch_size,aug_edge_weight,1.0,with_relation)
        
        degrees=nonzero_degrees+zero_degrees
        contrib_sum=nonzero_contrib_sum+zero_contrib_sum
        if div:
            degrees[degrees == 0] = 1
            res_emb = contrib_sum / degrees.unsqueeze(-1)
        else:
            res_emb = contrib_sum
        return res_emb
        
    

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(
        self, 
        embedding_size,
        channel,
        n_hops, 
        n_users,
        n_relations, 
        n_nodes,
        n_prefers,
        interact_mat,
        node_dropout_rate=0.5, 
        mess_dropout_rate=0.1,device=None
        ):
        super().__init__()
        
        #channel ---> embedding size
        self.device=device
        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users

        self.embedding_size = embedding_size
        self.channel = channel
        self.n_hops = n_hops
        self.n_nodes = n_nodes
        self.n_prefers = n_prefers
        self.interact_mat = interact_mat
        
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate


        initializer = nn.init.xavier_uniform_

        self.disen_weight_att = nn.Parameter(initializer(torch.empty(n_prefers, channel)))
        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)  # node dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)  # mess dropout
        
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        




    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb,  interact_mat,edge_index, edge_type,
               extra_edge_index,extra_edge_type,mess_dropout=True, node_dropout=False,drop_learn=False,method="add"):


        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling_torch(edge_index, edge_type, self.node_dropout_rate)
            extra_edge_index, extra_edge_type = self._edge_sampling_torch(extra_edge_index, extra_edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        aug_edge_weight,aug_extra_edge_weight=None,None
        node_emb  = torch.concat([user_emb,entity_emb],dim=0)     # [n_nodes, channel]


        aug_extra_edge_weight=self.calc_attn_score(node_emb,extra_edge_index,extra_edge_type).unsqueeze(-1)

        entity_res_emb = entity_emb                               # [n_entity, channel]
        
        node_res_emb = node_emb
        if method == "add":
            user_res_emb = user_emb
        elif method =="stack":
            user_res_emb = [user_emb]
        else:
            raise NotImplementedError


        for i in range(len(self.convs)):
            #all_emb,edge_index,edge_type,weight,aug_edge_weight=None
            entity_emb = self.convs[i](entity_emb,edge_index,edge_type-1,self.weight,aug_edge_weight,div=True)
            node_emb = self.convs[i](node_emb,extra_edge_index,extra_edge_type,self.extra_weight,
                                     aug_extra_edge_weight,div=True,with_relation=False)


            user_emb =torch.sparse.mm(interact_mat,entity_emb)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)

            entity_emb = F.normalize(entity_emb)
            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)
        


            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            node_res_emb = torch.add(node_res_emb, node_emb)
            if method == "add":
                user_res_emb = torch.add(user_res_emb, user_emb)
            elif method =="stack":
                user_res_emb +=[user_emb]
            else:
                raise NotImplementedError
            
        if method == "stack":
            user_res_emb =torch.stack(user_res_emb,dim=1)
            user_res_emb =torch.mean(user_res_emb,dim=1)


        gcn_res_emb=torch.concat([user_res_emb,entity_res_emb],dim=0) 
        return gcn_res_emb,node_res_emb



class ContrastiveLoss(nn.Module):
            """Contrastive Loss"""
    
            def __init__(self, temperature=0.5):
                super().__init__()
                self.temperature = temperature
    
            def forward(self, user_emb1, user_emb2, entity_emb1, entity_emb2):
                batch_size = user_emb1.shape[0]
                user_emb1 = F.normalize(user_emb1, dim=1)
                user_emb2 = F.normalize(user_emb2, dim=1)
                entity_emb1 = F.normalize(entity_emb1, dim=1)
                entity_emb2 = F.normalize(entity_emb2, dim=1)
    
                user_sim_matrix = torch.matmul(user_emb1, user_emb2.T) / self.temperature
                entity_sim_matrix = torch.matmul(entity_emb1, entity_emb2.T) / self.temperature
    
                user_labels = torch.arange(batch_size).to(self.device)
                entity_labels = torch.arange(entity_emb1.shape[0]).to(self.device)
    
                user_loss = F.cross_entropy(user_sim_matrix, user_labels) + F.cross_entropy(user_sim_matrix.T, user_labels)
                entity_loss = F.cross_entropy(entity_sim_matrix, entity_labels) + F.cross_entropy(entity_sim_matrix.T, entity_labels)
    
                loss = (user_loss + entity_loss) / 2
                return loss
    
    
    
class KPUN(KnowledgeRecommender):
    
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        #load parameters info
        self.embedding_size = config["embedding_size"]
        self.channel = config["channel"]
        self.n_hops = config["n_hops"]
        self.n_users = config["n_users"]
        self.n_relations = config["n_relations"] 
        self.n_prefers = config["n_prefers"]
        self.interact_mat = config["interact_mat"]
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        
        # define layers and loss
        self.n_nodes = self.n_users + self.n_entities
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.latent_embedding = nn.Embedding(self.n_factors, self.embedding_size)
        
        
        self.gcn = GraphConv(
            embedding_size = self.embedding_size,
            channel=self.channel,
            n_hops=self.n_hops,
            n_users=self.n_users,
            n_relations=self.n_relations,
            n_nodes=self.n_users + dataset.n_entities,
            n_prefers=self.n_prefers,
            interact_mat=self.interact_mat,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            device=self.device
        )
        
        self.mf_loss = BPRLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.restore_user_e = None
        self.restore_entity_e = None
        
        self.apply(xavier_uniform_initialization)




    
    def forward(self):
        user_emb = self.user_embedding.weight
        entity_emb = self.entity_embedding.weight
        user_gcn_emb, entity_gcn_emb, corr_loss = self.gcn(user_emb, entity_emb)
        return user_gcn_emb, entity_gcn_emb, corr_loss
    
    
    def calculate_loss(self, interaction):
        
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings, cor_loss = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        mf_loss = self.mf_loss(u_embeddings, pos_embeddings, neg_embeddings)
        if self.config['use_contrastive']:
            batch_size = self.config['batch_size']
            if batch_size > self.n_users:
                batch_size = self.n_users
            user_emb1 = user_all_embeddings[:self.n_users][:batch_size]
            user_emb2 = user_all_embeddings[:self.n_users][torch.randperm(self.n_users)][:batch_size]
            entity_emb1 = entity_all_embeddings[:self.n_entities][:batch_size]
            entity_emb2 = entity_all_embeddings[:self.n_entities][torch.randperm(self.n_entities)][:batch_size]
            cont_loss = self.contrastive_loss(user_emb1, user_emb2, entity_emb1, entity_emb2)
            loss = mf_loss + self.config['lambda_c'] * cont_loss + self.config['lambda_corr'] * cor_loss
        else:
            loss = mf_loss + self.config['lambda_corr'] * cor_loss
        return loss
    
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e, _ = self.forward()
            
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[item]
        scores = (u_embeddings * i_embeddings).sum(dim=1)
        return scores
    
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e, _ = self.forward()
            
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e
        scores = torch.matmul(u_embeddings, i_embeddings.t())
        return scores