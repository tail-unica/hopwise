# @Time   : 2020/10/2
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

"""SpectralCF
################################################

Reference:
    Lei Zheng et al. "Spectral collaborative filtering." in RecSys 2018.

Reference code:
    https://github.com/lzheng21/SpectralCF
"""

import torch

from hopwise.model.abstract_recommender import GeneralRecommender
from hopwise.model.init import xavier_uniform_initialization
from hopwise.model.loss import BPRLoss, EmbLoss
from hopwise.utils import InputType


class SpectralCF(GeneralRecommender):
    r"""SpectralCF is a spectral convolution model that directly learns latent factors of users and items
    from the spectral domain for recommendation.

    The spectral convolution operation with C input channels and F filters is shown as the following:

    .. math::
        \left[\begin{array} {c} X_{new}^{u} \\
        X_{new}^{i} \end{array}\right]=\sigma\left(\left(U U^{\top}+U \Lambda U^{\top}\right)
        \left[\begin{array}{c} X^{u} \\
        X^{i} \end{array}\right] \Theta^{\prime}\right)

    where :math:`X_{new}^{u} \in R^{n_{users} \times F}` and :math:`X_{new}^{i} \in R^{n_{items} \times F}`
    denote convolution results learned with F filters from the spectral domain for users and items, respectively;
    :math:`\sigma` denotes the logistic sigmoid function.

    Note:
        Our implementation is a improved version which is different from the original paper.
        For a better stability, we replace :math:`U U^T` with identity matrix :math:`I` and
        replace :math:`U \Lambda U^T` with laplace matrix :math:`L`.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.emb_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        # "A_hat = I + L" is equivalent to "A_hat = U U^T + U \Lambda U^T"
        I = dataset._create_eye_matrix()  # noqa: E741
        L = I - dataset._create_norm_adjacency_matrix(symmetric=False)
        A_hat = I + L
        self.A_hat = A_hat.to(self.device)

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_dim)
        self.filters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.normal(mean=0.01, std=0.02, size=(self.emb_dim, self.emb_dim)),
                    requires_grad=True,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            all_embeddings = self.sigmoid(torch.mm(all_embeddings, self.filters[k]))
            embeddings_list.append(all_embeddings)

        new_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(new_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
