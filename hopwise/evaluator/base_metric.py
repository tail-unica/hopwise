# @Time   : 2020/10/21
# @Author : Kaiyuan Li
# @email  : tsotfsk@outlook.com

# UPDATE
# @Time   : 2020/10/21, 2021/8/29
# @Author : Kaiyuan Li, Zhichao Feng
# @email  : tsotfsk@outlook.com, fzcbupt@gmail.com

"""hopwise.evaluator.abstract_metric
#####################################
"""

import numpy as np
import pandas as pd
import torch

from hopwise.utils import EvaluatorType


class AbstractMetric:
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.

    Args:
        config (Config): the config of evaluator.
    """

    smaller = False
    metric_need = []

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]

    def __init_subclass__(cls, **kwargs):
        """Automatically extend parent's metric_need if subclass defines metric_need."""
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "metric_need") and cls.metric_need is not cls.__bases__[0].metric_need:
            # Get parents' metric_need
            parent_metric_need = []
            for base in cls.__bases__:
                if hasattr(base, "metric_need"):
                    parent_metric_need.extend(base.metric_need)

            cls.metric_need = list(set(parent_metric_need + cls.metric_need))

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
        """
        raise NotImplementedError("Method [calculate_metric] should be implemented.")


class TopkMetric(AbstractMetric):
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.topk"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the bool matrix indicating whether the corresponding item is positive
        and number of positive items for each user.
        """
        rec_mat = dataobject.get("rec.topk")
        self.topk = dataobject.get("topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        return topk_idx.to(torch.bool).numpy(), pos_len_list.squeeze(-1).numpy()

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = f"{metric}@{k}"
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        """Calculate the value of the metric.

        Args:
            pos_index(numpy.ndarray): a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th \
            highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
            pos_len(numpy.ndarray): a vector representing the number of positive items per user, shape of ``(n_users,)``.

        Returns:
            numpy.ndarray: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
        """  # noqa: E501
        raise NotImplementedError("Method [metric_info] of top-k metric should be implemented.")


class LossMetric(AbstractMetric):
    """:class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
    implement an loss based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.VALUE
    metric_need = ["rec.score", "data.label"]

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        """Get scores that model predicted and the ground truth."""
        preds = dataobject.get("rec.score")
        trues = dataobject.get("data.label")

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        """Calculate the value of the metric.

        Args:
            preds (numpy.ndarray): the scores predicted by model, a one-dimensional vector.
            trues (numpy.ndarray): the label of items, which has the same shape as ``preds``.

        Returns:
            float: The value of the metric.
        """
        raise NotImplementedError("Method [metric_info] of loss-based metric should be implemented.")


class ConsumerTopKMetric(AbstractMetric):
    """:class:`ConsumerTopKMetric` is a base object of consumer-based metrics. If you want to
    implement a consumer-based metric, you can inherit this class.
    The consumer-based metrics are based on a binary partition of users and on the demographic parity notion,
    commonly measured as the absolute difference between the two groups in terms of a ranking metric.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["eval_data.user_feat", "rec.users"]
    smaller = True
    USER_GROUP_1 = 1
    USER_GROUP_2 = 2

    def __init__(self, config):
        super().__init__(config)
        self._ranking_metric = None
        self.sensitive_attribute = config["sensitive_attribute"]

        if self.sensitive_attribute is None:
            raise ValueError("The sensitive attribute is not specified in the config. Consumer metrics require it.")

    @property
    def ranking_metric(self):
        if self._ranking_metric is None:
            raise NotImplementedError("Use a subclass of ConsumerTopKMetric to calculate a specific ranking metric")
        return self._ranking_metric

    @ranking_metric.setter
    def ranking_metric(self, value):
        if not isinstance(value, TopkMetric):
            raise TypeError(f"The type of ranking_metric [{type(value)}] is not supported for ConsumerTopKMetric.")

        self._ranking_metric = value

    def get_group_mask(self, user_feat, interaction_users):
        # 0 is the padding sensitive attribute
        group1_mask = user_feat[self.sensitive_attribute][interaction_users] == self.USER_GROUP_1
        group2_mask = user_feat[self.sensitive_attribute][interaction_users] == self.USER_GROUP_2

        return group1_mask, group2_mask

    def used_info(self, dataobject):
        """Get the users features and the users in the interaction batch."""
        user_feat = dataobject.get("eval_data.user_feat")
        interaction_users = dataobject.get("rec.users")

        return self.get_group_mask(user_feat, interaction_users)

    def get_dp(self, result, group1_mask, group2_mask):
        """Get the absolute difference between the two groups in terms of a ranking metric.

        Args:
            group1_result (torch.Tensor): the result of the first group.
            group2_result (torch.Tensor): the result of the second group.

        Returns:
            torch.Tensor: the difference between the two groups in terms of a ranking metric.
        """
        group1_result = result[group1_mask, :].mean(axis=0, keepdims=True)
        group2_result = result[group2_mask, :].mean(axis=0, keepdims=True)

        return np.abs(group1_result - group2_result)

    def ranking_metric_info(self, pos_index, pos_len):
        raise NotImplementedError("Use a subclass of ConsumerTopKMetric to calculate a specific ranking metric")

    def calculate_metric(self, dataobject):
        group_mask1, group_mask2 = self.used_info(dataobject)
        pos_index, pos_len = self.ranking_metric.used_info(dataobject)
        ranking_result = self.ranking_metric_info(pos_index, pos_len)
        result = self.get_dp(ranking_result, group_mask1, group_mask2)
        metric_dict = self.ranking_metric.topk_result(self.__class__.__name__.lower(), result)
        return metric_dict


class PathQualityMetric(TopkMetric):
    # TODO: add support for gender and age based metrics
    """:class:`PathQualityMetric` is a base object of path-based metrics. If you want to
    implement a path based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.paths"]

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        paths = dataobject.get("rec.paths")
        return paths

    def normalized_ema(self, values):
        if max(values) == min(values):
            values = list(range(len(values)))

        values = pd.Series(values)
        ema_vals = values.ewm(span=len(values)).mean()
        normalized_ema_vals = (ema_vals - ema_vals.min()) / (ema_vals.max() - ema_vals.min())
        return normalized_ema_vals.to_numpy()
