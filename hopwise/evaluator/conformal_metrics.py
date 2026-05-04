# @Time   : 2026/03/01
# @Author : Alessandro Soccol
# @email  : alessandro.soccol@unica.it

"""hopwise.evaluator.conformal_metrics
#####################################
"""

import torch


class FNR:
    def __init__(self, config):
        self.topk = max(config["topk"])

        self.positive_u = None
        self.positive_i = None
        self._pos_index = None
        self._pos_len = None

    def set_positive_data(self, positive_u_list, positive_i_list):
        if isinstance(positive_u_list, (list, tuple)):
            self.positive_u = torch.cat(positive_u_list, dim=0)
        else:
            self.positive_u = positive_u_list

        if isinstance(positive_i_list, (list, tuple)):
            self.positive_i = torch.cat(positive_i_list, dim=0)
        else:
            self.positive_i = positive_i_list

    def set_metric_data(self, scores, topk_indices):
        pos_matrix = torch.zeros_like(scores, dtype=torch.int)
        pos_matrix[self.positive_u, self.positive_i] = 1
        pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_indices)
        matrix = torch.cat((pos_idx, pos_len_list), dim=1)
        topk_idx, pos_len_list = torch.split(matrix, [self.topk, 1], dim=1)
        self._pos_index = topk_idx.to(torch.bool)
        self._pos_len = pos_len_list.squeeze(-1)

    def calculate_metric(self):
        return 1 - (torch.cumsum(self._pos_index, dim=1) / self._pos_len.reshape(-1, 1))
