# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

# UPDATE
# @Time   : 2025
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""hopwise.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn.functional as F
from torch import nn


class SSMLoss(nn.Module):
    """Samples Softmax Loss (SSM) according to the implementation in"""

    def __init__(self, cosine_sim=True, temperature=1.0, eps=1e-7):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.temperature = temperature
        self.eps = eps

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        if self.cosine_sim:
            user_emb = nn.functional.normalize(user_emb, p=2, dim=-1)
            pos_item_emb = nn.functional.normalize(pos_item_emb, p=2, dim=-1)
            neg_item_emb = nn.functional.normalize(neg_item_emb, p=2, dim=-1)

        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1, keepdim=True)
        neg_score = torch.einsum("ijk, ik->ij", neg_item_emb, user_emb)

        # Temperatue-aware
        pos_score = torch.exp(pos_score / self.temperature)
        neg_score = torch.exp(neg_score / self.temperature).sum(dim=1, keepdim=True)

        total_score = pos_score + neg_score
        nce_loss = -(pos_score / total_score + self.eps).log().sum()

        return nce_loss


class SimCELoss(nn.Module):
    """Simplified Sampled Softmax Cross- Entropy Loss (SimCE),
    based on the implementation in https://arxiv.org/pdf/2406.16170"""

    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        # user_emb: [batch, dim]
        # pos_item_emb: [batch, dim]
        # neg_item_emb: [batch, num_neg, dim]
        num_neg, dim = neg_item_emb.shape[1], neg_item_emb.shape[2]
        neg_item_emb = neg_item_emb.reshape(-1, num_neg, dim)
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)
        neg_score = torch.max(neg_score, dim=-1).values
        loss = torch.relu(self.margin - pos_score + neg_score)

        return torch.mean(loss)


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super().__init__()

    def forward(self, parameters, reg_loss=None):
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super().__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class InnerProductLoss(nn.Module):
    r"""This is the inner-product loss used in CFKG for optimization."""

    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(-pos_score) + F.softplus(neg_score)).mean()


class LogisticLoss(nn.Module):
    """This is the logistic loss"""

    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, positive_score, negative_score, pos_regularization=None, neg_regularization=None):
        positive_labels = torch.ones_like(positive_score)
        negative_labels = -torch.ones_like(negative_score)

        positive_score = torch.mean(self.softplus(positive_score * positive_labels))
        negative_score = torch.mean(self.softplus(negative_score * negative_labels))

        if pos_regularization and neg_regularization:
            positive_score = positive_score + pos_regularization
            negative_score = negative_score + neg_regularization

        return torch.mean(positive_score + negative_score)
