# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/18
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""hopwise.evaluator.collector
################################################
"""

import copy

import torch

from hopwise.evaluator.register import Register, Register_KG
from hopwise.evaluator.utils import train_tsne


class DataStruct:
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value):
        if name not in self._data_dict:
            if isinstance(value, torch.Tensor):
                self._data_dict[name] = value.clone().detach()
            else:
                self._data_dict[name] = value
        # if not isinstance(self._data_dict[name], torch.Tensor):
        #     raise ValueError(f"{name} is not a tensor.")
        elif isinstance(self._data_dict[name], torch.Tensor):
            self._data_dict[name] = torch.cat((self._data_dict[name], value.clone().detach()), dim=0)
        else:
            self._data_dict[name] = self._data_dict[name] + value

    def __str__(self):
        data_info = "\nContaining:\n"
        for data_key in self._data_dict.keys():
            data_info += data_key + "\n"
        return data_info


class Collector:
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = "full" in config["eval_args"]["mode"]
        self.topk = self.config["topk"]
        self.device = self.config["device"]

    def train_data_collect(self, train_data):
        """Collect the evaluation resource from training data.

        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set("data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set("data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items", train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_users", train_data.dataset.user_counter)
        if self.register.need("data.history_index"):
            row = train_data.dataset.inter_feat[train_data.dataset.uid_field]
            col = train_data.dataset.inter_feat[train_data.dataset.iid_field]
            self.data_struct.set("data.history_index", torch.vstack([row, col]))
        if self.register.need("data.timestamp"):
            # timestamp don't match in .inter file.
            # this data should belong to the train set
            self.data_struct.set("data.timestamp", self.timestamp_dict(train_data))
        if self.register.need("data.max_path_type"):
            import numpy as np

            self.data_struct.set(
                "data.max_path_type", np.unique(train_data.dataset.kg_feat[train_data.dataset.relation_field])
            )
        if self.register.need("data.node_degree"):
            self.data_struct.set("data.node_degree", self.node_degree_dict(train_data))
        if self.register.need("data.max_path_length"):
            # how many path constraint are set?
            # 'path_constraint' is hardcoded and may change in the future
            self.data_struct.set("data.max_path_length", len(self.config["path_constraint"]))
        if self.register.need("data.rid2relation"):
            self.data_struct.set("data.rid2relation", train_data.dataset.field2id_token["relation_id"])

    def node_degree_dict(self, train_data):
        # from pgpr knowledge graph
        # https: // github.com/giacoballoccu/rep-path-reasoning-recsys/blob/main/models/PGPR/knowledge_graph.py
        aug_kg, _ = train_data.dataset.ckg_dict_graph()
        degrees = {}
        for etype in aug_kg.G:
            degrees[etype] = {}
            for eid in aug_kg.G[etype]:
                count = 0
                for r in aug_kg.G[etype][eid]:
                    count += len(aug_kg.G[etype][eid][r])
                degrees[etype][eid] = count
        return degrees

    def timestamp_dict(self, train_data):
        user_pid2timestamp = dict()

        users = train_data.dataset.inter_feat[train_data.dataset.uid_field]
        products = train_data.dataset.inter_feat[train_data.dataset.iid_field]
        timestamps = train_data.dataset.inter_feat[train_data.dataset.time_field]

        for user, product, timestamp in zip(users, products, timestamps):
            if user.item() not in user_pid2timestamp:
                user_pid2timestamp[user.item()] = list()
            user_pid2timestamp[user.item()].append((product.item(), timestamp.item()))

        return user_pid2timestamp

    def eval_data_collect(self, eval_data):
        """Collect the evaluation resource from evaluation data, such as user and item features.

        Args:
            eval_data (AbstractDataLoader): the evaluation dataloader which contains the evaluation data.

        """
        if self.register.need("eval_data.user_feat"):
            if not hasattr(eval_data.dataset, "user_feat") or eval_data.dataset.user_feat is None:
                raise AttributeError("Evaluation data does not include user features.")
            self.data_struct.set("eval_data.user_feat", eval_data.dataset.user_feat)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1).transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = 0.5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self,
        scores_tensor,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.

        Args:
            scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction(Interaction): batched eval data.
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """
        paths = None
        if isinstance(scores_tensor, tuple):
            scores_tensor, paths = scores_tensor

        if self.register.need("rec.users"):
            uid_field = self.config["USER_ID_FIELD"]
            self.data_struct.update_tensor("rec.users", interaction[uid_field])

        if self.register.need("rec.items"):
            # get topk
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            self.data_struct.update_tensor("rec.items", topk_idx)

        if self.register.need("rec.topk"):
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):
            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.meanrank", result)

        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", interaction[self.label_field].to(self.device))

        if self.register.need("data.test_batch_users"):
            self.data_struct.update_tensor("data.test_batch_users", scores_tensor.size(0))
        if self.register.need("rec.paths"):
            self.data_struct.update_tensor("rec.paths", paths)

    def model_collect(self, model: torch.nn.Module, load_best_model=False):
        """Collect the evaluation resource from model and do something with the model.

        Args:
            model (nn.Module): the trained recommendation model.
            load_best_model (bool): whether to load the best model.
        """

        if self.config["tsne"] is not None:
            train_tsne(model, self.config["tsne"], load_best_model)

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.

        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        for key in self.data_struct._data_dict:
            if isinstance(self.data_struct._data_dict[key], torch.Tensor):
                self.data_struct._data_dict[key] = self.data_struct._data_dict[key].cpu()
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label", "rec.paths"]:
            if key in self.data_struct:
                del self.data_struct[key]
        returned_struct.set("topk", self.topk)
        return returned_struct


class Collector_KG(Collector):
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        super().__init__(config)
        self.register = Register_KG(config)
        self.topk = self.config["topk_kg"]
