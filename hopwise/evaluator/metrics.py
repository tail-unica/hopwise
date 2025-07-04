# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2020/08/12, 2021/8/29, 2020/9/16, 2021/7/2
# @Author  :   Kaiyuan Li, Zhichao Feng, Xingyu Pan, Zihan Lin
# @email   :   tsotfsk@outlook.com, fzcbupt@gmail.com, panxy@ruc.edu.cn, zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2025
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

r"""hopwise.evaluator.metrics
############################

Suppose there is a set of :math:`n` items to be ranked. Given a user :math:`u` in the user set :math:`U`,
we use :math:`\hat R(u)` to represent a ranked list of items that a model produces, and :math:`R(u)` to
represent a ground-truth set of items that user :math:`u` has interacted with. For top-k recommendation, only
top-ranked items are important to consider. Therefore, in top-k evaluation scenarios, we truncate the
recommendation list with a length :math:`K`. Besides, in loss-based metrics, :math:`S` represents the
set of user(u)-item(i) pairs, :math:`\hat r_{u i}` represents the score predicted by the model,
:math:`{r}_{u i}` represents the ground-truth labels.

"""

import inspect
import sys
from collections import Counter
from logging import getLogger

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

from hopwise.evaluator.base_metric import AbstractMetric, ConsumerTopKMetric, LossMetric, PathQualityMetric, TopkMetric
from hopwise.evaluator.utils import _binary_clf_curve
from hopwise.utils import EvaluatorType

# TopK Metrics


class Hit(TopkMetric):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
    you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
    we call it a hit.

    .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    .. math::
        \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

    :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
    :math:`\emptyset` denotes the empty set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("hit", result)
        return metric_dict

    def metric_info(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int)


class MRR(TopkMetric):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
    of the first relevant item found by an algorithm.

    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    .. math::
       \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

    :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("mrr", result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float64)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopkMetric):
    r"""MAP_ (also known as Mean Average Precision) is meant to calculate
    average precision for the relevant items.

    Note:
        In this case the normalization factor used is :math:`\frac{1}{min(|\hat R(u)|, K)}`, which prevents your
        AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture
        all the correct ones.

    .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
       \mathrm{MAP@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{min(|\hat R(u)|, K)} \sum_{j=1}^{|\hat{R}(u)|} I\left(\hat{R}_{j}(u) \in R(u)\right) \cdot  Precision@j)

    :math:`\hat{R}_{j}(u)` is the j-th item in the recommendation list of \hat R (u)).
    """  # noqa: E501

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("map", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float64), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float64)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class Recall(TopkMetric):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
       \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

    :math:`|R(u)|` represents the item count of :math:`R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("recall", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float64)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float64)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result


class Precision(TopkMetric):
    r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
    out of all the recommended items. We average the metric for each user :math:`u` get the final result.

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}

    :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("precision", result)
        return metric_dict

    def metric_info(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


# CTR Metrics


class GAUC(AbstractMetric):
    r"""GAUC (also known as Grouped Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve grouped by user. We weighted the index of each user :math:`u` by the number of positive
    samples of users to get the final result.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3219819.3219823>`__

    Note:
        It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
        It is also not limited to k. Due to our padding for `scores_tensor` with `-np.inf`, the padding
        value will influence the ranks of origin items. Therefore, we use descending sort here and make
        an identity transformation  to the formula of `AUC`, which is shown in `auc_` function.
        For readability, we didn't do simplification in the code.

    .. math::
        \begin{align*}
            \mathrm {AUC(u)} &= \frac {{{|R(u)|} \times {(n+1)} - \frac{|R(u)| \times (|R(u)|+1)}{2}} -
            \sum\limits_{i=1}^{|R(u)|} rank_{i}} {{|R(u)|} \times {(n - |R(u)|)}} \\
            \mathrm{GAUC} &= \frac{1}{\sum_{u \in U} |R(u)|}\sum_{u \in U} |R(u)| \cdot(\mathrm {AUC(u)})
        \end{align*}

    :math:`rank_i` is the descending rank of the i-th items in :math:`R(u)`.
    """  # noqa: E501

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.meanrank"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        mean_rank = dataobject.get("rec.meanrank").numpy()
        pos_rank_sum, user_len_list, pos_len_list = np.split(mean_rank, 3, axis=1)
        user_len_list, pos_len_list = (
            user_len_list.squeeze(-1),
            pos_len_list.squeeze(-1),
        )
        result = self.metric_info(pos_rank_sum, user_len_list, pos_len_list)
        return {"gauc": round(result, self.decimal_place)}

    def metric_info(self, pos_rank_sum, user_len_list, pos_len_list):
        """Get the value of GAUC metric.

        Args:
            pos_rank_sum (numpy.ndarray): sum of descending rankings for positive items of each users.
            user_len_list (numpy.ndarray): the number of predicted items for users.
            pos_len_list (numpy.ndarray): the number of positive items for users.

        Returns:
            float: The value of the GAUC.
        """
        neg_len_list = user_len_list - pos_len_list
        # check positive and negative samples
        any_without_pos = np.any(pos_len_list == 0)
        any_without_neg = np.any(neg_len_list == 0)
        non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)
        if any_without_pos:
            logger = getLogger()
            logger.warning(
                "No positive samples in some users, "
                "true positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= pos_len_list != 0
        if any_without_neg:
            logger = getLogger()
            logger.warning(
                "No negative samples in some users, "
                "false positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= neg_len_list != 0
        if any_without_pos or any_without_neg:
            item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
            user_len_list, neg_len_list, pos_len_list, pos_rank_sum = map(lambda x: x[non_zero_idx], item_list)

        pair_num = (
            (user_len_list + 1) * pos_len_list - pos_len_list * (pos_len_list + 1) / 2 - np.squeeze(pos_rank_sum)
        )
        user_auc = pair_num / (neg_len_list * pos_len_list)
        result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
        return result


class AUC(LossMetric):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve.

    .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users. We call the interface
        in `scikit-learn`, and code calculates the metric using the variation of following formula.

    .. math::
        \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
        \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

    :math:`M` denotes the number of positive items.
    :math:`N` denotes the total number of user-item interactions.
    :math:`rank_i` denotes the descending rank of the i-th positive item.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("auc", dataobject)

    def metric_info(self, preds, trues):
        fps, tps = _binary_clf_curve(trues, preds)
        if len(fps) > 2:  # noqa: PLR2004
            optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if fps[-1] <= 0:
            logger = getLogger()
            logger.warning("No negative samples in y_true, false positive value should be meaningless")
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            logger = getLogger()
            logger.warning("No positive samples in y_true, true positive value should be meaningless")
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        result = sk_auc(fpr, tpr)
        return result


# Loss-based Metrics


class MAE(LossMetric):
    r"""MAE_ (also known as Mean Absolute Error regression loss) is used to evaluate the difference between
    the score predicted by the model and the actual behavior of the user.

    .. _MAE: https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math::
        \mathrm{MAE}=\frac{1}{|{S}|} \sum_{(u, i) \in {S}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`|S|` represents the number of pairs in :math:`S`.
    """

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("mae", dataobject)

    def metric_info(self, preds, trues):
        return mean_absolute_error(trues, preds)


class RMSE(LossMetric):
    r"""RMSE_ (also known as Root Mean Squared Error) is another error metric like `MAE`.

    .. _RMSE: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math::
       \mathrm{RMSE} = \sqrt{\frac{1}{|{S}|} \sum_{(u, i) \in {S}}(\hat{r}_{u i}-r_{u i})^{2}}
    """

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("rmse", dataobject)

    def metric_info(self, preds, trues):
        return np.sqrt(mean_squared_error(trues, preds))


class LogLoss(LossMetric):
    r"""Logloss_ (also known as logistic loss or cross-entropy loss) is used to evaluate the probabilistic
    output of the two-class classifier.

    .. _Logloss: http://wiki.fast.ai/index.php/Log_Loss

    .. math::
        LogLoss = \frac{1}{|S|} \sum_{(u,i) \in S}(-((r_{u i} \ \log{\hat{r}_{u i}}) + {(1 - r_{u i})}\ \log{(1 - \hat{r}_{u i})}))
    """  # noqa: E501

    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("logloss", dataobject)

    def metric_info(self, preds, trues):
        eps = 1e-15
        preds = np.float64(preds)
        preds = np.clip(preds, eps, 1 - eps)
        loss = np.sum(-trues * np.log(preds) - (1 - trues) * np.log(1 - preds))
        return loss / len(preds)


class ItemCoverage(AbstractMetric):
    r"""ItemCoverage_ computes the coverage of recommended items over all items.

    .. _ItemCoverage: https://en.wikipedia.org/wiki/Coverage_(information_systems)

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/1864708.1864761>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
       \mathrm{Coverage@K}=\frac{\left| \bigcup_{u \in U} \hat{R}(u) \right|}{|I|}
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get("rec.items")
        num_items = dataobject.get("data.num_items")
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("itemcoverage", k)
            metric_dict[key] = round(self.get_coverage(item_matrix[:, :k], num_items), self.decimal_place)
        return metric_dict

    def get_coverage(self, item_matrix, num_items):
        """Get the coverage of recommended items over all items

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the `coverage` metric.
        """
        unique_count = np.unique(item_matrix).shape[0]
        return unique_count / num_items


class AveragePopularity(AbstractMetric):
    r"""AveragePopularity computes the average popularity of recommended items.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
        \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

    :math:`\phi(i)` is the number of interaction of item i in training data.
    """

    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and the popularity of items in training data"""
        item_counter = dataobject.get("data.count_items")
        item_matrix = dataobject.get("rec.items")
        return item_matrix.numpy(), dict(item_counter)

    def calculate_metric(self, dataobject):
        item_matrix, item_count = self.used_info(dataobject)
        result = self.metric_info(self.get_pop(item_matrix, item_count))
        metric_dict = self.topk_result("averagepopularity", result)
        return metric_dict

    def get_pop(self, item_matrix, item_count):
        """Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            item_count(dict): the number of interaction of items in training data.

        Returns:
            numpy.ndarray: the popularity of items in the recommended list.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form

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


class ShannonEntropy(AbstractMetric):
    r"""ShannonEntropy_ presents the diversity of the recommendation items.
    It is the entropy over items' distribution.

    .. _ShannonEntropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__

    .. math::
        \mathrm {ShannonEntropy@K}=-\sum_{i=1}^{|I|} p(i) \log p(i)

    :math:`p(i)` is the probability of recommending item i
    which is the number of item i in recommended list over all items.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items."""
        item_matrix = dataobject.get("rec.items")
        return item_matrix.numpy()

    def calculate_metric(self, dataobject):
        item_matrix = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("shannonentropy", k)
            metric_dict[key] = round(self.get_entropy(item_matrix[:, :k]), self.decimal_place)
        return metric_dict

    def get_entropy(self, item_matrix):
        """Get shannon entropy through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.

        Returns:
            float: the shannon entropy.
        """
        item_count = dict(Counter(item_matrix.flatten()))
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        result = 0.0
        for cnt in item_count.values():
            p = cnt / total_num
            result += -p * np.log(p)
        return result / len(item_count)


class GiniIndex(AbstractMetric):
    r"""GiniIndex presents the diversity of the recommendation items.
    It is used to measure the inequality of a distribution.

    .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3308560.3317303>`__.

    .. math::
        \mathrm {GiniIndex@K}=\left(\frac{\sum_{i=1}^{|I|}(2 i-|I|-1) P{(i)}}{|I| \sum_{i=1}^{|I|} P{(i)}}\right)

    :math:`P{(i)}` represents the number of times all items appearing in the recommended list,
    which is indexed in non-decreasing order (P_{(i)} \leq P_{(i+1)}).
    """

    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ["rec.items", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get("rec.items")
        num_items = dataobject.get("data.num_items")
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = "{}@{}".format("giniindex", k)
            metric_dict[key] = round(self.get_gini(item_matrix[:, :k], num_items), self.decimal_place)
        return metric_dict

    def get_gini(self, item_matrix, num_items):
        """Get gini index through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the gini index.
        """
        item_count = dict(Counter(item_matrix.flatten()))
        sorted_count = np.array(sorted(item_count.values()))
        num_recommended_items = sorted_count.shape[0]
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
        gini_index /= num_items
        return gini_index


class TailPercentage(AbstractMetric):
    r"""TailPercentage_ computes the percentage of long-tail items in recommendation items.

    .. _TailPercentage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

    For further details, please refer to the `paper <https://arxiv.org/pdf/2007.12329.pdf>`__.

    .. math::
        \mathrm {TailPercentage@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_{u}} {\delta(i \in T)}}{|R_{u}|}

    :math:`\delta(·)` is an indicator function.
    :math:`T` is the set of long-tail items,
    which is a portion of items that appear in training data seldomly.

    Note:
        If you want to use this metric, please set the parameter 'tail_ratio' in the config
        which can be an integer or a float in (0,1]. Otherwise it will default to 0.1.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        self.tail = config["tail_ratio"]
        if self.tail is None or self.tail <= 0:
            self.tail = 0.1

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set."""
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_tail(self, item_matrix, count_items):
        """Get long-tail percentage through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            count_items(dict): the number of interaction of items in training data.

        Returns:
            float: long-tail percentage.
        """
        if self.tail > 1:
            tail_items = [item for item, cnt in count_items.items() if cnt <= self.tail]
        else:
            count_items = sorted(count_items.items(), key=lambda kv: (kv[1], kv[0]))
            cut = max(int(len(count_items) * self.tail), 1)
            count_items = count_items[:cut]
            tail_items = [item for item, cnt in count_items]
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = 1 if row[j] in tail_items else 0
        return value

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        result = self.metric_info(self.get_tail(item_matrix, count_items))
        metric_dict = self.topk_result("tailpercentage", result)
        return metric_dict

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

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


# Consumer Metrics dynamic creation


def create_consumer_metric_class(topk_metric):
    """Dynamically creates and returns a new consumer class with the given name, base classes, and attributes."""
    topk_metric_class = getattr(sys.modules[__name__], topk_metric)
    signature = inspect.signature(topk_metric_class.metric_info)
    parameters = signature.parameters

    if "pos_len" in parameters:

        def ranking_metric_info(self, pos_index, pos_len):
            return self.ranking_metric.metric_info(pos_index, pos_len)
    else:

        def ranking_metric_info(self, pos_index, pos_len):
            return self.ranking_metric.metric_info(pos_index)

    consumer_metric_class_name = f"Delta{topk_metric}"
    consumer_metric_class = type(
        consumer_metric_class_name, (ConsumerTopKMetric,), {"ranking_metric_info": ranking_metric_info}
    )
    consumer_metric_class.metric_need.extend(
        [need for need in topk_metric_class.metric_need if need not in consumer_metric_class.metric_need]
    )

    def factory_init(self, config):
        super(consumer_metric_class, self).__init__(config)
        self.ranking_metric = topk_metric_class(config)

    consumer_metric_class.__init__ = factory_init

    return consumer_metric_class


DeltaHit = create_consumer_metric_class("Hit")
DeltaMRR = create_consumer_metric_class("MRR")
DeltaMAP = create_consumer_metric_class("MAP")
DeltaNDCG = create_consumer_metric_class("NDCG")
DeltaPrecision = create_consumer_metric_class("Precision")
DeltaRecall = create_consumer_metric_class("Recall")

# Beyond Accuracy Metrics


class Serendipity(AbstractMetric):
    r"""Serendipity is a measure of how surprising the recommended items are to the user.
    It is defined as the fraction of recommended items that are not popular in the training data.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.num_items", "data.num_users", "data.count_items", "data.history_index"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and the popularity of items in training data"""
        num_users = dataobject.get("data.num_users")
        num_items = dataobject.get("data.num_items")
        item_counter = dataobject.get("data.count_items")
        item_matrix = dataobject.get("rec.items")
        history_matrix = dataobject.get("data.history_index")

        items, count = zip(*item_counter.items())
        item_counter = np.zeros(num_items, dtype=np.int)
        item_counter[list(items)] = count

        return item_matrix.numpy(), item_counter, history_matrix, num_users

    def calculate_metric(self, dataobject):
        item_matrix, item_count, history_matrix, num_users = self.used_info(dataobject)
        pop_recs = np.tile(item_count, (num_users, 1))
        pop_recs[tuple(history_matrix)] = 0
        pop_recs = pop_recs[1:]  # remove the padding
        pop_topk = np.argsort(pop_recs, axis=-1)[:, ::-1][:, : max(self.topk)]
        topk_intersection = np.zeros_like(item_matrix, dtype=bool)
        for u, (user_topk, user_pop_topk) in enumerate(zip(item_matrix, pop_topk)):
            topk_intersection[u] = np.isin(user_topk, user_pop_topk)

        result = self.metric_info(topk_intersection)
        metric_dict = self.topk_result("serendipity", result)
        return metric_dict

    def metric_info(self, values):
        return 1 - (values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1))

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


class Novelty(AbstractMetric):
    """
    Paper:
    Novelty: Inverse of popularity of the items recommended to the user
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        num_items = dataobject.get("data.num_items")
        return item_matrix.numpy(), dict(count_items), int(num_items)

    def get_pop(self, item_matrix, item_count):
        """Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            item_count(dict): the number of interaction of items in training data.

        Returns:
            numpy.ndarray: the popularity of items in the recommended list.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def normalize_popularity(self, item_matrix, pop_matrix, item_count, num_items):
        normalized_item_count = dict()
        min_pop = min(item_count.values()) if len(item_count.values()) == num_items else 0
        max_pop = max(item_count.values())

        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j, item in enumerate(row):
                if item not in normalized_item_count:
                    normalized_item_count[item] = (pop_matrix[i, j] - min_pop) / (max_pop - min_pop)
        return normalized_item_count

    def calculate_metric(self, dataobject):
        item_matrix, item_count, num_items = self.used_info(dataobject)
        pop_matrix = self.get_pop(item_matrix, item_count)
        normalized_item_count = self.normalize_popularity(item_matrix, pop_matrix, item_count, num_items)
        metric_dict = {}
        for k in self.topk:
            novelty_score = []
            for topk_user in item_matrix:
                novelty_items_topk = [1 - normalized_item_count[iid] for iid in topk_user]
                novelty_score.append(np.mean(novelty_items_topk))
            metric_dict[f"novelty@{k}"] = round(np.mean(novelty_score), self.decimal_place)
        return metric_dict


# Perceived Path Explanation Quality


class LIR(PathQualityMetric):
    r"""Linking Interaction Recency (LIR)

    This property serves to quantify the time since the linking interaction in the explanation path occurred.
    Given a user :math:`u \in U` and the set :math:`P_u` of items this user interacted with,
    we denote the list of their interactions, sorted chronologically, by :math:`T_u = [(p^i, t^i)]`,
    where :math:`p^i \in P_u` is a item experienced by the user, :math:`t^i \in \mathbb N`
    is the timestamp that interaction occurred, and :math:`t^i \leq t^{i+1}` :math:`\forall i = 1, \dots, |P_u|`.

    They applied an exponentially weighed moving average to the timestamps included in :math:`T_u`,
    to obtain the LIR of each interaction performed by the user :math:`u`.
    Specifically, given an interaction :math:`(p^i, t^i) \in T_u`, the LIR for that interaction
    was computed as follows:

    .. math::
        \mathrm{LIR(p^i, t^i)} = ( 1 - \beta_{LIR} ) \cdot LIR(p^{i-1}, t^{i-1}) + \beta_{LIR} \cdot t^i


    where :math:`\beta_{LIR} \in [0, 1]` is a decay associated to the interaction time,
    and :math:`LIR(p^1, t^1) = t^1`.
    The LIR values were min-max normalized for each user to lay in the range :math:`[0, 1]`,
    with values close to 0 (1) meaning that the linking interaction is far away (recent) in time.
    In the case of a recommended item not being present in the user's interaction history,
    the LIR of the linking interaction was set to 0, meaning that the linking interaction
    is very recent (i.e., the user has just interacted with the item).
    The overall LIR for explanations in a recommended list was obtained
    by averaging the LIR of the linking interactions for the selected explanation path of each recommended item.

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3477495.3532041>`.
    """

    metric_need = ["data.timestamp", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.li_idx = 1  # The linking interaction is the second element in the path tuple

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        timestamp_matrix = dataobject.get("data.timestamp")
        num_items = dataobject.get("data.num_items")

        lir_matrix = self.get_lir_matrix(timestamp_matrix)
        result = self.metric_info(lir_matrix, paths, num_items)
        metric_dict = self.topk_result("lir", result)
        return metric_dict

    def get_lir_matrix(self, timestamp_matrix):
        lir_matrix = np.zeros_like(timestamp_matrix, dtype=np.float32)
        for uid, user_inter_timestamp in enumerate(timestamp_matrix):
            inter_mask = user_inter_timestamp > 0
            if not inter_mask.any():
                continue

            sort_idxs = np.argsort(user_inter_timestamp[inter_mask])
            sorted_timestamps = user_inter_timestamp[inter_mask][sort_idxs]
            ema_timestamps = self.normalized_ema(sorted_timestamps)
            if not np.isnan(ema_timestamps).any():
                inter_indices = np.where(inter_mask)[0][sort_idxs]
                lir_matrix[uid, inter_indices] = ema_timestamps

        return lir_matrix

    def metric_info(self, lir_matrix, paths, num_items):
        users, li_items = [], []
        for user, _, _, path in paths:
            is_item = path[self.li_idx][1] == "item" or (
                path[self.li_idx][1] == "entity" and path[self.li_idx][-1] < num_items
            )
            if is_item:
                users.append(user)
                li_items.append(int(path[self.li_idx][-1]))

        return lir_matrix[users, li_items]

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class Fidelity(PathQualityMetric):
    """
    Fidelity
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        result = self.metric_info(paths)
        metric_dict = self.topk_result("Fidelity", result)
        return metric_dict

    def metric_info(self, paths):
        user_paths = dict()
        for user, item, _, _ in paths:
            if user not in user_paths:
                user_paths[user] = set()
            user_paths[user].add(item)

        path_topk_len = []
        for items in user_paths.values():
            path_topk_len.append(len(items))

        return np.array(path_topk_len)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to
            `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        for k in self.topk:
            key = f"{metric}@{k}"
            avg_result = min((value / k).mean(axis=0), 1.0)
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class SEP(PathQualityMetric):
    r"""Popularity of Shared Entity (SEP)

    This property serves to quantify the extent to which the shared entity included in an explanation-path is popular.
    They assume that the number of relationships a shared entity is involved in the KG is a proxy of its popularity.
    For instance, the popularity of an actor is computed by counting how many movies that actor starred in.
    They denote the list of entities of a given type $\lambda$ in the KG, sorted based on their popularity,
    by :math:`_\lambda = [(e^i, v^i)]`, where :math:`e^i \in E_\lambda` is an entity
    of type :math:`\lambda`, :math:`v^i \in \mathbb N` is the number of relationships a shared entity is involved in
    (in-degree),
    and :math:`v^i \leq v^{i+1}` :math:`\forall i = 1, \dots, |E_\lambda|`.
    We applied an exponential decay to the popularity scores in :math:`S_\lambda$, to get the SEP of an entity
    of type :math:`\lambda`, as:

    .. math::
        \mathrm{SEP(e^i, v^i)} = ( 1 - \beta_{SEP} ) \cdot SEP(e^{i-1}, v^{i-1}) + \beta_{SEP} \cdot v^i

    where :math:`\beta_{SEP}` is a decay related to the popularity, and :math:`SEP(e^1, v^1) = v^1`
    The SEP values w ere min-max normalized for each entity type to lay in the range [0, 1],
    with values close to 0 (1) when the entity has a low (high) popularity.
    The shared entity novelty might be obtained as :math:`SEN(e^i, v^i) = 1-SEP(e^i, v^i).

    The overall SEP for explanations in a recommended list was obtained by averaging the SEP
    values of the shared entity # noqa: E501
    in the selected path for each recommended item.

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3477495.3532041>`.
    """

    metric_need = ["data.node_degree"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        node_degree = dataobject.get("data.node_degree")

        sep_matrix = self.get_sep_matrix(node_degree)
        result = self.metric_info(paths, sep_matrix)
        metric_dict = self.topk_result("SEP", result)
        return metric_dict

    def get_sep_matrix(self, node_degree):
        # Precompute entity distribution
        sep_matrix = {}

        for node_type, eid_degree in node_degree.items():
            eid_degree_tuples = list(zip(eid_degree.keys(), eid_degree.values()))
            eid_degree_tuples.sort(key=lambda x: x[1])
            ema_es = self.normalized_ema([x[1] for x in eid_degree_tuples])
            iid_weight = {}
            for idx in range(len(ema_es)):
                iid = eid_degree_tuples[idx][0]
                iid_weight[iid] = ema_es[idx]

            sep_matrix[node_type] = iid_weight
        return sep_matrix

    def metric_info(self, paths, sep_matrix):
        seps_topk = []
        for _, _, _, path in paths:
            shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
            if shared_entity_type == "item":
                shared_entity_type = "entity"
            seps_topk.append(sep_matrix[shared_entity_type][shared_entity_id])
        if not seps_topk:
            return 0.0
        return np.array(seps_topk)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = f"{metric}@{k}"
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class LID(PathQualityMetric):
    """
    Diversity of Linked Interaction (LID)

    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        result = self.metric_info(paths)
        metric_dict = self.topk_result("LID", result)
        return metric_dict

    def metric_info(self, paths):
        unique_linking_interaction = dict()
        for user, _, _, path in paths:
            linked_interaction_id = path[1][-1]
            if user not in unique_linking_interaction:
                unique_linking_interaction[user] = [0, set()]
            unique_linking_interaction[user][0] += 1  # number of path for that specific user
            unique_linking_interaction[user][1].add(linked_interaction_id)

        lid = []
        for user in unique_linking_interaction:
            n_user_paths = unique_linking_interaction[user][0]
            li = unique_linking_interaction[user][1]
            if not n_user_paths:
                lid.append(0.0)
                continue
            lid.append(len(li) / n_user_paths)

        return np.array(lid)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class SED(PathQualityMetric):
    """
    Diversity of Shared Entities

    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)

        result = self.metric_info(paths)
        metric_dict = self.topk_result("SED", result)
        return metric_dict

    def metric_info(self, paths):
        unique_shared_entity = dict()
        for user, _, _, path in paths:
            shared_entity_id = path[-2][-1]
            if user not in unique_shared_entity:
                unique_shared_entity[user] = [0, set()]
            # number of path for that specific user
            unique_shared_entity[user][0] += 1
            unique_shared_entity[user][1].add(shared_entity_id)

        sed = []
        for user in unique_shared_entity:
            n_user_paths = unique_shared_entity[user][0]
            se = unique_shared_entity[user][1]
            if not n_user_paths:
                sed.append(0.0)
                continue
            sed.append(len(se) / n_user_paths)

        return np.array(list(sed))

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class PTD(PathQualityMetric):
    """
    Path Type Diversity
    """

    metric_need = ["data.max_path_type"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        max_path_type = dataobject.get("data.max_path_type")

        result = self.metric_info(paths, max_path_type)
        metric_dict = self.topk_result("PTD", result)
        return metric_dict

    def metric_info(self, paths, max_path_type):
        unique_path_type = dict()
        for user, _, _, path in paths:
            # Get path type
            path_type = path[-1][0]
            if path_type == "self_loop":  # Handle size 3
                path_type = path[-2][0]
            # Track number of paths for each user and seen path types
            if user not in unique_path_type:
                unique_path_type[user] = [0, set()]
            # number of path for that specific user
            unique_path_type[user][0] += 1
            unique_path_type[user][1].add(path_type)

        ptd = []
        for user in unique_path_type:
            n_user_paths = unique_path_type[user][0]
            pt = unique_path_type[user][1]
            if not n_user_paths:
                ptd.append(0.0)
                continue
            ptd.append(len(pt) / min(n_user_paths, len(max_path_type)))

        return np.array(ptd)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class PTC(PathQualityMetric):
    """
    Path Type Concentration (PTC)
    """

    metric_need = ["data.max_path_type"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        max_path_type = dataobject.get("data.max_path_type")

        result = self.metric_info(paths, max_path_type)
        metric_dict = self.topk_result("PTC", result)
        return metric_dict

    def metric_info(self, paths, max_path_type):
        user_simpson_index = dict()
        for user, _, _, path in paths:
            if user not in user_simpson_index:
                # 0 is N, the second is n_path_for_patterns
                user_simpson_index[user] = [0, {k: 0 for k in set(max_path_type)}]
            #  Get path type
            path_type = path[-1][0]

            if path_type == "self_loop":  # Handle size 3
                path_type = path[-2][0]

            if path_type not in user_simpson_index[user][1]:
                user_simpson_index[user][1][path_type] = 0

            user_simpson_index[user][1][path_type] += 1
            user_simpson_index[user][0] += 1

        ptc = []
        for user in user_simpson_index:
            numerator = 0
            for path_type, n_path_type_ith in user_simpson_index[user][1].items():
                numerator += n_path_type_ith * (n_path_type_ith - 1)

            if user_simpson_index[user][0] * (user_simpson_index[user][0] - 1) == 0:
                ptc.append(0)
                continue

            ptc.append(1 - (numerator / (user_simpson_index[user][0] * (user_simpson_index[user][0] - 1))))

        return np.array(ptc)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class PPT(PathQualityMetric):
    """
    Path Pattern Type
    """

    metric_need = ["data.max_path_length", "data.rid2relation"]

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        max_path_pattern = dataobject.get("data.max_path_length")

        rid2r_name = {i: rel for i, rel in enumerate(dataobject.get("data.rid2relation"))}
        result = self.metric_info(paths, max_path_pattern, rid2r_name)
        metric_dict = self.topk_result("PPT", result)
        return metric_dict

    def metric_info(self, paths, max_path_pattern, rid2r_name):
        unique_path_pattern = dict()
        for user, _, _, path in paths:
            if user not in unique_path_pattern:
                unique_path_pattern[user] = [0, set()]
            path_pattern = [rid2r_name[path_tuple[0]] for path_tuple in path[1:]]
            unique_path_pattern[user][0] += 1
            unique_path_pattern[user][1].add("_".join(path_pattern))

        ppt = []
        for user in unique_path_pattern:
            n_paths = unique_path_pattern[user][0]
            n_path_patterns = len(unique_path_pattern[user][1])
            ppt.append(min(n_path_patterns / min(n_paths, max_path_pattern), 1.0))

        return np.array(ppt)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class LITD(PathQualityMetric):
    """
    Linked Interaction Type Diversity
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        result = self.metric_info(paths)
        metric_dict = self.topk_result("LITD", result)
        return metric_dict

    def metric_info(self, paths):
        unique_linking_interaction_types = dict()
        for user, _, _, path in paths:
            if user not in unique_linking_interaction_types:
                unique_linking_interaction_types[user] = [0, set()]
            linked_interaction_type = path[1][1]
            unique_linking_interaction_types[user][0] += 1
            unique_linking_interaction_types[user][1].add(linked_interaction_type)

        litd = []
        for user in unique_linking_interaction_types:
            n_paths = unique_linking_interaction_types[user][0]
            n_linked_interaction_types = len(unique_linking_interaction_types[user][1])
            litd.append(n_linked_interaction_types / n_paths)
        return np.array(litd)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict


class SETD(PathQualityMetric):
    """
    Shared Entities Type Diversity
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        paths = self.used_info(dataobject)
        result = self.metric_info(paths)
        metric_dict = self.topk_result("SETD", result)
        return metric_dict

    def metric_info(self, paths):
        unique_shared_entity_type = dict()
        for user, _, _, path in paths:
            if user not in unique_shared_entity_type:
                unique_shared_entity_type[user] = [0, set()]
            shared_entity_type = path[-2][1]
            unique_shared_entity_type[user][0] += 1
            unique_shared_entity_type[user][1].add(shared_entity_type)

        setd = []
        for user in unique_shared_entity_type:
            n_paths = unique_shared_entity_type[user][0]
            n_shared_entity_types = len(unique_shared_entity_type[user][1])
            setd.append(n_shared_entity_types / n_paths)
        return np.array(setd)

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
            metric_dict[key] = round(avg_result, self.decimal_place)
        return metric_dict
