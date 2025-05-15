# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import ones_
import gc

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from sklearn.metrics import *

import logging
from tqdm import tqdm  # Importa tqdm per la barra di caricamento
import os
from time import time

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.utils import InputType


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*out_features, out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        
        # N, e_num, dim
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W) # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We) # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size()) # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1) # (N, e_num, 2*out_features)

class KGEncoder(nn.Module):

    def __init__(self, config, dataset, kg_dataset):
        super().__init__()

        self.user_history_dict = dict()
        for user_id, item_id in zip(dataset.inter_feat['user_id'].numpy(), dataset.inter_feat['item_id'].numpy()):
            if user_id not in self.user_history_dict:
                self.user_history_dict[user_id] = []
            self.user_history_dict[user_id].append(item_id)

        self.maxhis = config['maxhis']
        self.batch_size = config['batch_size']
        self.kgcn = config['kgcn']
        self.dropout = config['dropout']
        self.keep_prob = 1 - self.dropout #Added
        self.A_split = config['A_split']

        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.dataset = dataset
        self.kg_dataset = kg_dataset
        self.gat = GAT(self.latent_dim, self.latent_dim,
                       dropout=0.4, alpha=0.2).train()
        
        self.config = config
        self.process_inter_feat("train")
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.dataset.user_num, self.dataset.item_num))
        self.__init_weight()
    
    def __init_weight(self):
        self.num_users = self.dataset.user_num
        self.num_items = self.dataset.item_num
        self.num_entities = self.dataset.entity_num
        self.num_relations = self.dataset.relation_num
       
        # Print with colors the stats
        #logging.info(f'\033[36mtrainSize\033[0m: \033[35m{self.dataset.trainSize}\033[0m') TODO capire come ottenere trainSize
        logging.info(f'\033[36mn_user\033[0m: \033[35m{self.dataset.user_num}\033[0m')
        logging.info(f'\033[36mm_item\033[0m: \033[35m{self.dataset.item_num}\033[0m')

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # item and kg entity
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        # relation weights
        self.W_R = nn.Parameter(torch.Tensor(
            self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        logging.info('use NORMAL distribution UI')
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        logging.info('use NORMAL distribution ENTITY')
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)
        
        self.f = nn.Sigmoid()
        #self.Graph = self.dataset.getSparseGraph()
        self.Graph = self.getSparseGraph() # TODO: Change this, because return a GraphDict and not a Tensor...(?)
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        self.kg_dict, self.item2relations = self.get_kg_dict(
            self.num_items)
        
    def process_inter_feat(self, filetype):
        logging.info("\033[92mProcessing interaction features...\033[0m")
        inter_feat = self.dataset.inter_feat
        UniqueUsers, Item, User = [], [], []
        UsersHis = []
        dataSize = 0

        user_ids = inter_feat['user_id'].numpy()
        item_ids = inter_feat['item_id'].numpy()

        last_uid = None
        user_items = []

        for uid, iid in zip(user_ids, item_ids):
            if filetype == "train":
                if last_uid is None or uid != last_uid:
                    user_items = []
                this_his = user_items[-self.maxhis:] if self.maxhis > 0 else user_items[:]
                this_his += [-1] * (self.maxhis - len(this_his))
                UsersHis.append(this_his)
                user_items.append(iid)
            UniqueUsers.append(uid)
            User.append(uid)
            Item.append(iid)
            self.m_item = max(self.dataset.item_num, iid)
            self.n_user = max(self.dataset.user_num, uid)
            dataSize += 1
            last_uid = uid

        setattr(self, f'{filetype}UniqueUsers', np.array(UniqueUsers))
        setattr(self, f'{filetype}User', np.array(User))
        setattr(self, f'{filetype}Item', np.array(Item))
        setattr(self, f'{filetype}UsersHis', np.array(UsersHis))
        setattr(self, f'{filetype}Size', dataSize)

        logging.info(f"{dataSize} interactions for {filetype}")


    def get_inetrfeat(self, filetype):
        """
        Adattato per Hopwise: Legge i dati di interazione da inter_feat e popola gli attributi necessari.

        Args:
            filetype (str): Tipo di file da leggere ('train', 'test', 'valid').
        """
        if not hasattr(self.dataset, 'inter_feat'):
            logging.info(f"\033[91mDataset does not contain inter_feat\033[0m")
            return

        inter_feat = self.dataset.inter_feat

        maxhis = self.config["maxhis"]

        # Estrai USER_ID e ITEM_ID
        user_col = self.dataset.uid_field  # Nome della colonna per USER_ID
        item_col = self.dataset.iid_field  # Nome della colonna per ITEM_ID

        if user_col not in inter_feat or item_col not in inter_feat:
            logging.info(f"\033[91mColumns {user_col} or {item_col} not found in inter_feat\033[0m")
            return

        UniqueUsers, Item, User = [], [], []
        UsersHis = []
        dataSize = 0

        # Itera sui dati di inter_feat
        user_ids = inter_feat[user_col]
        item_ids = inter_feat[item_col]

        for uid, iid in zip(user_ids, item_ids):
            uid = int(uid)  # ID utente
            iid = int(iid)  # ID oggetto

            if filetype == "train":
                # Costruzione della cronologia per gli utenti
                his = []
                for i in range(len(item_ids)):
                    this_his = item_ids[:i] if maxhis <= 0 else item_ids[max(0, i - maxhis):i]
                    padding = torch.full((maxhis - this_his.size(0),), -1, dtype=this_his.dtype, device=this_his.device)
                    this_his = torch.cat((this_his, padding))
                    his.append(this_his)
                UsersHis.extend(his)

            UniqueUsers.append(uid)
            User.append(uid)
            Item.append(iid)

            self.m_item = max(self.dataset.item_num, iid)
            self.n_user = max(self.dataset.user_num, uid)
            dataSize += 1

        # Salva i dati come attributi della classe
        setattr(self, f'{filetype}UniqueUsers', np.array(UniqueUsers))
        setattr(self, f'{filetype}User', np.array(User))
        setattr(self, f'{filetype}Item', np.array(Item))
        if filetype == "train":
            setattr(self, f'{filetype}UsersHis', np.array(UsersHis))
        setattr(self, f'{filetype}Size', dataSize)

        logging.info(f"{dataSize} interactions for {filetype}")

    def get_kg_dict(self, item_num):
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dataset.get(item, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) > self.dataset.entity_num):
                    i2es[item] = torch.IntTensor(tails).cuda()[:self.dataset.entity_num]
                    i2rs[item] = torch.IntTensor(relations).cuda()[:self.dataset.entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.dataset.entity_count]*(self.dataset.entity_num-len(tails)))
                    relations.extend([self.dataset.relation_count]*(self.dataset.entity_num-len(relations)))
                    i2es[item] = torch.IntTensor(tails).cuda()
                    i2rs[item] = torch.IntTensor(relations).cuda()
            else:
                i2es[item] = torch.IntTensor([self.dataset.item_num]*self.dataset.entity_num).cuda()
                i2rs[item] = torch.IntTensor([self.dataset.relation_num]*self.dataset.entity_num).cuda()
        return i2es, i2rs
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        '''Calculate the connection graph in graph convolution, including A~, etc.
        The returned data is a processed list, the length of the list is determined by self.fold, 
        and each item in the list is a sparse matrix representing the connection matrix of the entity 
        at the corresponding index for that length.
        '''        
        logging.info("generating adjacency matrix")
        s = time()
        # The rows and columns of adj_mat concatenate users and items, marking known connections.
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok() # Convert the matrix to dictionary format (key-value pairs)
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv) # 对角线元素为每行求和
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        logging.info(f"costing {end-s}s, saved norm_mat...")

        self.dataset.G = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.dataset.G = self.dataset.G.coalesce().cuda()
        logging.info("\033[31mDidn't split the matrix\033[0m")
        return self.dataset.G

    def computer(self):
        with torch.no_grad():
            users_emb = self.embedding_user.weight
            items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            if self.dropout:
                if self.training:
                    g_droped = self.__dropout(self.keep_prob)
                else:
                    g_droped = self.Graph
            else:
                g_droped = self.Graph

            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        if (self.kgcn == "GAT"):
            return self.cal_item_embedding_gat(kg)
        elif self.kgcn == "RGAT":
            return self.cal_item_embedding_rgat(kg)
        elif (self.kgcn == "MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif (self.kgcn == "NO"):
            return self.embedding_item.weight

    # def cal_item_embedding_gat(self, kg: dict):
        
        # print("\033[92mStarting with batches...\033[0m")
        # item_embs = self.embedding_item(torch.IntTensor(
            # list(kg.keys())).cuda())  # item_num, emb_dim
        # item_num, entity_num_each
        # item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        # entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        # padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            # item_entities), torch.zeros_like(item_entities)).float()
        # return self.gat(item_embs, entity_embs, padding_mask)
    
    def cal_item_embedding_gat(self, kg: dict):
        batch_size = self.batch_size
        item_keys = list(kg.keys())
        item_embs_list = []
        
        for i in range(0, len(item_keys), batch_size):
            batch_keys = item_keys[i:i+batch_size]
            item_embs = self.embedding_item(torch.IntTensor(batch_keys).cuda())  # batch_size, emb_dim
            # batch_size, entity_num_each
            item_entities = torch.stack([kg[key] for key in batch_keys])
            # batch_size, entity_num_each, emb_dim
            entity_embs = self.embedding_entity(item_entities)
            # batch_size, entity_num_each
            padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
            batch_embs = self.gat(item_embs, entity_embs, padding_mask)
            item_embs_list.append(batch_embs)
        
        return torch.cat(item_embs_list, dim=0)

    def cal_item_embedding_rgat(self, kg: dict):
        batch_size = self.batch_size
        item_keys = list(kg.keys())
        item_embs_list = []

        for i in range(0, len(item_keys), batch_size):
            batch_keys = item_keys[i:i+batch_size]

            # Pulisci cache prima di ogni batch
            gc.collect()
            torch.cuda.empty_cache()

            item_embs = self.embedding_item(torch.IntTensor(batch_keys).cuda())
            item_entities = torch.stack([kg[key] for key in batch_keys])
            item_relations = torch.stack([self.item2relations[key] for key in batch_keys])

            entity_embs = self.embedding_entity(item_entities)
            relation_embs = self.embedding_relation(item_relations)

            padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
                item_entities), torch.zeros_like(item_entities)).float()

            with torch.no_grad():
                batch_embs = self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

            item_embs_list.append(batch_embs)

        return torch.cat(item_embs_list, dim=0)

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).cuda())  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * \
            padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / \
            padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs+entity_embs_mean


class KGLRR(KnowledgeRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

        # Load Full Knowledge Graph in dict form
        if dataset.G is not None and dataset.kg_relation is not None:
            self.G, self.kg_dataset = dataset.G, dataset.kg_dataset
        else:
            # Load the knowledge graph from the dataset
            print("\033[92mLoading KG...\033[0m")
            self.G, self.kg_dataset = dataset.ckg_dict_graph()
            # self.G, self.kg_dataset = dataset.ckg_graph("pyg")
            if config["save_dataset"]:
                dataset.save()
        
        self.encoder = KGEncoder(config, dataset, self.kg_dataset)
        self.latent_dim = config['latent_dim_rec']

        self.r_logic = config['r_logic']
        self.r_length = config['r_length']
        self.layers = config['layers']
        self.sim_scale = config['sim_scale']
        self.loss_sum = config['loss_sum']
        self.l2s_weight = config['l2_loss']
        
        self.dataset = dataset
        self.num_items = dataset.item_num
        

        self._init_weights()
        self.bceloss = nn.BCEWithLogitsLoss()

    def _init_weights(self):
        self.true = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 1, size=[1, self.latent_dim]).astype(np.float32)).cuda(), requires_grad=False)

        self.and_layer = torch.nn.Linear(self.latent_dim * 2, self.latent_dim)
        for i in range(self.layers):
            setattr(self, 'and_layer_%d' % i, torch.nn.Linear(self.latent_dim * 2, self.latent_dim * 2))

        self.or_layer = torch.nn.Linear(self.latent_dim * 2, self.latent_dim)
        for i in range(self.layers):
            setattr(self, 'or_layer_%d' % i, torch.nn.Linear(self.latent_dim * 2, self.latent_dim * 2))

    def logic_or(self, vector1, vector2, train=False):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'or_layer_%d' % i)(vector))
        vector = self.or_layer(vector)
        return vector
    
    def logic_and(self, vector1, vector2, train=False):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'and_layer_%d' % i)(vector))
        vector = self.and_layer(vector)
        return vector

    def logic_regularizer(self, train:bool, check_list:list, constraint, constraint_valid):
        # This function calculates the gap between logical expressions and the real world

        # length
        r_length = constraint.norm(dim=2).sum()
        check_list.append(('r_length', r_length))
        
        # and
        r_and_true = 1 - self.similarity(self.logic_and(constraint, self.true, train=train), constraint)
        r_and_true = (r_and_true * constraint_valid).sum()
        check_list.append(('r_and_true', r_and_true))

        r_and_self = 1 - self.similarity(self.logic_and(constraint, constraint, train=train), constraint)
        r_and_self = (r_and_self * constraint_valid).sum()
        check_list.append(('r_and_self', r_and_self))
        
        # or
        r_or_true = 1 - self.similarity(self.logic_or(constraint, self.true, train=train), self.true)
        r_or_true = (r_or_true * constraint_valid).sum()
        check_list.append(('r_or_true', r_or_true))
        
        r_or_self = 1 - self.similarity(self.logic_or(constraint, constraint, train=train), constraint)
        r_or_self = (r_or_self * constraint_valid).sum()
        check_list.append(('r_or_self', r_or_self))

        r_loss = r_and_true + r_and_self \
                    + r_or_true + r_or_self
        
        if self.r_logic > 0:
            r_loss = r_loss * self.r_logic
        else:
            r_loss = torch.from_numpy(np.array(0.0, dtype=np.float32)).cuda()
            r_loss.requires_grad = True

        r_loss += r_length * self.r_length
        check_list.append(('r_loss', r_loss))
        return r_loss

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    def uniform_size(self, vector1, vector2, train=False):
        # Removed vector size normalization
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        elif vector2.size() != vector1.size():
            vector2 = vector2.expand_as(vector1)
        if train:
            r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
            r12 = r12.cuda().unsqueeze(-1)
            new_v1 = r12 * vector1 + (-r12 + 1) * vector2
            new_v2 = r12 * vector2 + (-r12 + 1) * vector1
            return new_v1, new_v2
        return vector1, vector2

    def predict(self, interaction, explain=True):
        users = interaction[self.USER_ID]
        bs = users.size(0)
        maxhis = self.encoder.maxhis

        history_list = []

        for uid in users.tolist():
            user_his = self.encoder.user_history_dict.get(uid, [])
            user_his = user_his[-maxhis:]  # limita alla lunghezza massima
            user_his += [-1] * (maxhis - len(user_his))  # padding
            history_list.append(user_his)

        history = torch.tensor(history_list, dtype=torch.long, device=users.device)  # [B, H]
        item_embed = self.encoder.computer()[1]   # item_num * V
        
        his_valid = history.ge(0).float()  # B * H

        maxlen = int(his_valid.sum(dim=1).max().item())
        
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V

        tmp_o = None
        for i in range(maxlen):
            tmp_o_valid = his_valid[:, i].unsqueeze(-1)
            if tmp_o is None:
                tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
            else:
                # 有valid标志才能运算or，否则就是历史长度没有这么长（才会不valid），此时直接保持原本的内容不变
                tmp_o = self.logic_or(tmp_o, elements[:, i, :]) * tmp_o_valid + \
                        tmp_o * (-tmp_o_valid + 1)  # B * V
        or_vector = tmp_o  # B * V
        left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1
        
        prediction = []
        for i in range(bs):
            sent_vector = left_valid[i] * self.logic_and(or_vector[i].unsqueeze(0
                                                            ).repeat(self.num_items, 1), item_embed) \
                            + (-left_valid[i] + 1) * item_embed  # item_size * V
            ithpred = self.similarity(sent_vector, self.true, sigmoid=True)  # item_size
            prediction.append(ithpred)

        prediction = torch.stack(prediction).cuda()

        # if explain:
        #     explaination = self.explain(users, history, prediction)
        #     return prediction, explaination
        return prediction
    
    def full_sort_predict(self, interaction):
        r"""Full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users, n_candidate_items]
        """
        # La funzione predict già fa quello che serve (utenti vs tutti gli item)
        prediction = self.predict(interaction, explain=False)  # [batch_size, num_items]
        return prediction

    
    def predict_or_and(self, users, pos, neg, history):
        # 存储用于检查的内容：逻辑正则化 
        # 对嵌入计算L2正则化
        check_list = []
        bs = users.size(0)
        users_embed, item_embed = self.encoder.computer()

        # 历史数据中每个商品都为正标记，但是历史后段可能取-1表示没有这么长
        his_valid = history.ge(0).float()  # B * H
        maxlen = int(his_valid.sum(dim=1).max().item())
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V
        
        # 用于之后的验证，每个向量都应满足逻辑表达式中的相应约束，valid表示限制向量中对应元的有效性
        constraint = [elements.view([bs, -1, self.latent_dim])]  # B * H * V
        constraint_valid = [his_valid.view([bs, -1])]  # B * H
        
        tmp_o = None
        for i in range(maxlen):
            tmp_o_valid = his_valid[:, i].unsqueeze(-1)
            if tmp_o is None:
                tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
            else:
                # 有valid标志才能运算or，否则就是历史长度没有这么长（才会不valid），此时直接保持原本的内容不变
                tmp_o = self.logic_or(tmp_o, elements[:, i, :]) * tmp_o_valid + \
                        tmp_o * (-tmp_o_valid + 1)  # B * V
                constraint.append(tmp_o.view([bs, 1, self.latent_dim]))  # B * 1 * V
                constraint_valid.append(tmp_o_valid)  # B * 1
        or_vector = tmp_o  # B * V
        left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1

        right_vector_true = item_embed[pos]  # B * V
        right_vector_false = item_embed[neg]  # B * V
        
        constraint.append(right_vector_true.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(torch.ones((bs,1), device='cuda'))  # B * 1   # 表示所有将要判断的item都是有效的
        constraint.append(right_vector_false.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(torch.ones((bs,1), device='cuda'))  # B * 1

        sent_vector = self.logic_and(or_vector, right_vector_true) * left_valid \
                      + (-left_valid + 1) * right_vector_true  # B * V
        constraint.append(sent_vector.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1
        prediction_true = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])  # B
        check_list.append(('prediction_true', prediction_true))

        sent_vector = self.logic_and(or_vector, right_vector_false) * left_valid \
                      + (-left_valid + 1) * right_vector_false  # B * V
        constraint.append(sent_vector.view([bs, 1, self.latent_dim]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1
        prediction_false = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])  # B
        check_list.append(('prediction_false', prediction_false))

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)

        return prediction_true, prediction_false, check_list, constraint, constraint_valid

    def calculate_loss(self, interaction):
        """
        Calculates the total loss by combining:
        - BCE Loss (rloss)
        - Entropy Loss (tloss)
        - L2 Loss (l2loss)
        """
        # Estrazione dei tensori dal dizionario interaction
        batch_users = interaction['user_id'].long().cuda()
        batch_pos = interaction['item_id'].long().cuda()
        batch_neg = interaction['neg_item_id'].long().cuda()

        # Costruzione della history nello stesso modo di predict
        bs = batch_users.size(0)
        maxhis = self.encoder.maxhis
        history_list = []

        for uid in batch_users.tolist():
            user_his = self.encoder.user_history_dict.get(uid, [])
            user_his = user_his[-maxhis:]  # limita alla lunghezza massima
            user_his += [-1] * (maxhis - len(user_his))  # padding
            history_list.append(user_his)

        batch_history = torch.tensor(history_list, dtype=torch.long, device=batch_users.device)

        # Forward del modello con le 3 componenti della loss
        rloss, tloss, l2loss = self.forward(False, 0, batch_users, batch_pos, batch_neg, batch_history)

        # Combinazione delle 3 componenti
        total_loss = rloss + tloss + l2loss

        return total_loss


    def triple_loss(self, TItemScore, FItemScore):
        bce_loss = self.bceloss(TItemScore.sigmoid(), torch.ones_like(TItemScore)) + \
                    self.bceloss(FItemScore.sigmoid(), torch.zeros_like(FItemScore))
        # Input positive and negative example scores, maximizing the score difference
        if self.loss_sum == 1:
            loss = torch.sum(
            torch.nn.functional.softplus(-(TItemScore - FItemScore)))
        else:
            loss = torch.mean(
            torch.nn.functional.softplus(-(TItemScore - FItemScore)))
        return (loss+bce_loss)*0.5

    def l2_loss(self, users, pos, neg, history):
        users_embed, item_embed = self.encoder.computer()
        users_emb = users_embed[users]
        pos_emb = item_embed[pos]
        neg_emb = item_embed[neg]
        his_valid = history.ge(0).float()  # B * H
        elements = item_embed[history.abs()] * his_valid.unsqueeze(-1)  # B * H * V
        # L2 regularization loss
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2) +
                          elements.norm(2).pow(2))/float(len(users))
        if self.loss_sum == 0:
            reg_loss /= users.size(0)
        return reg_loss * self.l2s_weight

    def check(self, check_list):
        logging.info(os.linesep)
        for t in check_list:
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

    def forward(self, print_check:bool, return_pred:bool, *args, **kwards):
        prediction1, prediction0, check_list, constraint, constraint_valid = self.predict_or_and(*args, **kwards)
        rloss = self.logic_regularizer(False, check_list, constraint, constraint_valid)
        tloss = self.triple_loss(prediction1, prediction0)
        l2loss = self.l2_loss(*args, **kwards)

        if print_check:
            self.check(check_list)

        if return_pred:
            return prediction1, rloss+tloss+l2loss
        return rloss, tloss, l2loss


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
