# @Time   : 2024/11/24
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""ConvKB
##################################################
Reference:
    Nguyen et al. "A Novel Embedding Model for Knowledge Base Completion Based on
    Convolutional Neural Network." in NAACL 2018.

Reference code:
    https://github.com/torchkge-team/torchkge
"""

import torch
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.model.init import xavier_normal_initialization
from hopwise.model.loss import LogisticLoss, RegLoss
from hopwise.utils import InputType


class ConvKB(KnowledgeRecommender):
    r"""ConvKB: The main differences from ConvE are that when scoring h, r and t,
    it concatenates them into a d x 3 matrix. This output undergoes convolution
    by a set of omega of T filters of shape 1x3, resulting in a Tx3 feature map.
    This feature map goes through a dense layer with one neuron and weights W, resulting in the final score.

    Note:
        In this version, we sample recommender data and knowledge data separately, and put them together for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.out_channels = config["out_channels"]
        self.kernel_size = config["kernel_size"]
        self.drop_prob = config["dropout_prob"]
        self.lmbda = config["lambda"]
        self.ui_relation = dataset.field2token_id["relation_id"][dataset.ui_relation]

        # Embeddings and Layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.out_channels, (self.kernel_size, 3))
        self.conv2_bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(self.drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.embedding_size - self.kernel_size + 1) * self.out_channels, 1, bias=False)

        # Loss and Regularization
        self.loss = LogisticLoss()
        self.reg = RegLoss()

        # Embeddings Initialization
        self.apply(xavier_normal_initialization)

    def forward(self, head, relation, tail):
        h = head.unsqueeze(1)
        r = relation.unsqueeze(1)
        t = tail.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.embedding_size - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return score

    def _get_regularization(self, head, relation, tail):
        l2_reg = torch.mean(head**2) + torch.mean(tail**2) + torch.mean(relation**2)
        l2_reg = self.reg(self.conv_layer.parameters(), l2_reg)
        l2_reg = self.reg(self.fc_layer.parameters(), l2_reg)
        return self.lmbda * l2_reg

    def _get_rec_embeddings(self, user, positive_items, negative_items):
        relation_users = torch.tensor([self.ui_relation] * user.shape[0], device=self.device)
        h = self.user_embedding(user)
        r = self.relation_embedding(relation_users)
        t_pos = self.entity_embedding(positive_items)
        t_neg = self.entity_embedding(negative_items)
        return h, r, t_pos, t_neg

    def _get_kg_embeddings(self, head, relation, positive_tails, negative_tails):
        h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        t_pos = self.entity_embedding(positive_tails)
        t_neg = self.entity_embedding(negative_tails)
        return h, r, t_pos, t_neg

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]

        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        head = interaction[self.HEAD_ENTITY_ID]

        relation = interaction[self.RELATION_ID]

        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        users_embedding, relations_user_embedding, pos_items_embedding, neg_items_embedding = self._get_rec_embeddings(
            user, pos_item, neg_item
        )
        heads_embedding, relations_kg_embedding, pos_tails_embedding, neg_tails_embedding = self._get_kg_embeddings(
            head, relation, pos_tail, neg_tail
        )

        score_pos_users = self.forward(users_embedding, relations_user_embedding, pos_items_embedding)
        score_neg_users = self.forward(users_embedding, relations_user_embedding, neg_items_embedding)
        score_pos_kg = self.forward(heads_embedding, relations_kg_embedding, pos_tails_embedding)
        score_neg_kg = self.forward(heads_embedding, relations_kg_embedding, neg_tails_embedding)

        pos_users_reg = self._get_regularization(users_embedding, relations_user_embedding, pos_items_embedding)
        neg_users_reg = self._get_regularization(users_embedding, relations_user_embedding, neg_items_embedding)
        pos_kg_reg = self._get_regularization(heads_embedding, relations_kg_embedding, pos_tails_embedding)
        neg_kg_reg = self._get_regularization(heads_embedding, relations_kg_embedding, neg_tails_embedding)

        rec_loss = self.loss(-score_pos_users, -score_neg_users, pos_users_reg, neg_users_reg)
        kg_loss = self.loss(-score_pos_kg, -score_neg_kg, pos_kg_reg, neg_kg_reg)
        return rec_loss + kg_loss

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        relations = torch.tensor([self.ui_relation] * users.shape[0], device=self.device)

        users_e = self.user_embedding(users)
        relations_e = self.relation_embedding(relations)
        items_e = self.entity_embedding(items)

        return self.forward(users_e, relations_e, items_e)

    def predict_kg(self, interaction):
        heads = interaction[self.HEAD_ENTITY_ID]
        relations = interaction[self.RELATION_ID]
        tails = interaction[self.TAIL_ENTITY_ID]

        heads_e = self.entity_embedding(heads)
        relations_e = self.relation_embedding(relations)
        tails_e = self.entity_embedding(tails)

        return self.forward(heads_e, relations_e, tails_e)
