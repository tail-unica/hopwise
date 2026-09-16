# @Time    :   2020/11/1
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :    2021/7/2, 2021/7/5
# @Author  :    Zihan Lin, Zhichao Feng
# @email   :    zhlin@ruc.edu.cn, fzcbupt@gmail.com

import os
import sys
import unittest

sys.path.append(os.getcwd())
import numpy as np
import pytest

from hopwise.config import Config
from hopwise.evaluator.register import metrics_dict

parameters_dict = {
    "topk": [10],
    "metric_decimal_place": 4,
}

config = Config("BPR", "ml-1m", config_dict=parameters_dict)
pos_idx = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
    ]
)
pos_len = np.array([1, 3, 4, 2])

item_matrix = np.array([[5, 7, 3], [4, 5, 2], [2, 3, 5], [1, 4, 6], [5, 3, 7]])

num_items = 8

item_count = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


class TestTopKMetrics(unittest.TestCase):
    def test_hit(self):
        name = "hit"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1]]).tolist(),
        )

    def test_ndcg(self):
        name = "ndcg"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [
                        1,
                        (1 / np.log2(2) / (1 / np.log2(2) + 1 / np.log2(3))),
                        ((1 / np.log2(2) + 1 / np.log2(4)) / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))),
                    ],
                    [0, 0, (1 / np.log2(4) / (1 / np.log2(2) + 1 / np.log2(3)))],
                ]
            ).tolist(),
        )

    def test_mrr(self):
        name = "mrr"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1 / 3]]).tolist(),
        )

    def test_map(self):
        name = "map"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, (1 / 2), (1 / 3) * ((1 / 1) + (2 / 3))],
                    [0, 0, (1 / 3) * (1 / 2)],
                ]
            ).tolist(),
        )

    def test_recall(self):
        name = "recall"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx, pos_len).tolist(),
            np.array([[0, 0, 0], [1 / 3, 2 / 3, 3 / 3], [1 / 4, 1 / 4, 2 / 4], [0, 0, 1 / 2]]).tolist(),
        )

    def test_precision(self):
        name = "precision"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(pos_idx).tolist(),
            np.array([[0, 0, 0], [1 / 1, 2 / 2, 3 / 3], [1 / 1, 1 / 2, 2 / 3], [0, 0, 1 / 3]]).tolist(),
        )

    def test_itemcoverage(self):
        name = "itemcoverage"
        Metric = metrics_dict[name](config)
        self.assertEqual(Metric.get_coverage(item_matrix, num_items), 7 / 8)

    def test_averagepopularity(self):
        name = "averagepopularity"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(Metric.get_pop(item_matrix, item_count)).tolist(),
            np.array(
                [
                    [4 / 1, 4 / 2, 6 / 3],
                    [3 / 1, 7 / 2, 8 / 3],
                    [1 / 1, 3 / 2, 7 / 3],
                    [0 / 1, 3 / 2, 8 / 3],
                    [4 / 1, 6 / 2, 6 / 3],
                ]
            ).tolist(),
        )

    def test_giniindex(self):
        name = "giniindex"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.get_gini(item_matrix, num_items),
            ((-7) * 0 + (-5) * 1 + (-3) * 1 + (-1) * 2 + 1 * 2 + 3 * 2 + 5 * 3 + 7 * 4) / (8 * (3 * 5)),
        )

    def test_shannonentropy(self):
        name = "shannonentropy"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.get_entropy(item_matrix),
            -np.mean(
                [
                    1 / 15 * np.log(1 / 15),
                    2 / 15 * np.log(2 / 15),
                    3 / 15 * np.log(3 / 15),
                    2 / 15 * np.log(2 / 15),
                    4 / 15 * np.log(4 / 15),
                    1 / 15 * np.log(1 / 15),
                    2 / 15 * np.log(2 / 15),
                ]
            ),
        )

    def test_tailpercentage(self):
        name = "tailpercentage"
        Metric = metrics_dict[name](config)
        self.assertEqual(
            Metric.metric_info(Metric.get_tail(item_matrix, item_count)).tolist(),
            np.array(
                [
                    [0 / 1, 0 / 2, 0 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                    [1 / 1, 1 / 2, 1 / 3],
                    [0 / 1, 0 / 2, 0 / 3],
                ]
            ).tolist(),
        )


# -------------------------------------------------------------------------
# Beyond-utility metrics
# -------------------------------------------------------------------------

BEYOND_UTILITY_TOL = 1e-4
BEYOND_UTILITY_NUM_ITEMS = 10
NO_HISTORY = np.empty((2, 0), dtype=int)


def _beyond_utility_config(topk):
    return Config("BPR", "ml-1m", config_dict={"topk": topk, "metric_decimal_place": 4})


@pytest.fixture(scope="module")
def beyond_utility_config():
    return _beyond_utility_config([5])


class _NumpyWrap:
    """Mimics the tensor stored under ``rec.items``."""

    def __init__(self, arr):
        self._arr = np.array(arr, dtype=int)

    def numpy(self):
        return self._arr


class _DataObject:
    """Minimal stand-in for the evaluator data struct, exposing only the keys a metric reads."""

    def __init__(self, store):
        self._store = store

    def get(self, key):
        return self._store[key]


# item 0 is the most popular, item 9 the least popular
SERENDIPITY_COUNT_ITEMS = {i: BEYOND_UTILITY_NUM_ITEMS - i for i in range(BEYOND_UTILITY_NUM_ITEMS)}

SERENDIPITY_CASES = [
    pytest.param([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], NO_HISTORY, 0.0, id="all_recs_popular"),
    pytest.param([[9, 8, 7, 6, 5], [9, 8, 7, 6, 5]], NO_HISTORY, 1.0, id="no_recs_popular"),
    pytest.param([[0, 1, 2, 9, 8], [0, 1, 2, 9, 8]], NO_HISTORY, 1 - 3 / 5, id="three_of_five_popular"),
    pytest.param(
        # history_index is (user_ids; item_ids), users 1-indexed due to padding.
        # Items {0, 1} are removed from both users' popularity ranking, so only item 2 stays popular.
        [[0, 1, 2, 9, 8], [0, 1, 2, 9, 8]],
        np.array([[1, 1, 2, 2], [0, 1, 0, 1]], dtype=int),
        1 - 1 / 5,
        id="history_items_are_not_popular",
    ),
    pytest.param([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]], NO_HISTORY, (0.0 + 1.0) / 2, id="two_users_mean_over_users"),
]


def _serendipity(config, rec_items, history_index):
    store = {
        "rec.items": _NumpyWrap(rec_items),
        "data.count_items": SERENDIPITY_COUNT_ITEMS,
        "data.num_items": BEYOND_UTILITY_NUM_ITEMS,
        "data.num_users": len(rec_items) + 1,  # padding row 0
        "data.history_index": history_index,
    }
    return metrics_dict["serendipity"](config).calculate_metric(_DataObject(store))


@pytest.mark.parametrize(("rec_items", "history_index", "expected"), SERENDIPITY_CASES)
def test_serendipity(beyond_utility_config, rec_items, history_index, expected):
    """Serendipity@k(u) = 1 - |top-k recs of u in the k most popular non-history items| / k, averaged."""
    result = _serendipity(beyond_utility_config, rec_items, history_index)
    assert float(result["serendipity@5"]) == pytest.approx(expected, abs=BEYOND_UTILITY_TOL)


def test_serendipity_depends_on_k():
    result = _serendipity(_beyond_utility_config([2, 5]), [[3, 4, 9, 8, 7]], NO_HISTORY)
    # @2: popular {0, 1}, no overlap -> 1.0; @5: popular {0..4}, overlap {3, 4} -> 1 - 2/5
    got = {k: float(result[f"serendipity@{k}"]) for k in (2, 5)}
    assert got == pytest.approx({2: 1.0, 5: 1 - 2 / 5}, abs=BEYOND_UTILITY_TOL)


# item 0 is the least popular (count 1), item 9 the most popular (count 10)
NOVELTY_COUNT_ITEMS = {i: i + 1 for i in range(BEYOND_UTILITY_NUM_ITEMS)}


def _item_novelty(i):
    # min-max normalized popularity is (count - 1) / (10 - 1) = i / 9
    return 1.0 - i / 9


def _mean_novelty(items):
    return sum(_item_novelty(i) for i in items) / len(items)


NOVELTY_CASES = [
    pytest.param([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], _mean_novelty([0, 1, 2, 3, 4]), id="low_popularity_items"),
    pytest.param([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]], _mean_novelty([5, 6, 7, 8, 9]), id="high_popularity_items"),
    pytest.param([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]], _mean_novelty([2, 3, 4, 5, 6]), id="mid_popularity_items"),
    pytest.param(
        [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]],
        (_mean_novelty([0, 1, 2, 3, 4]) + _mean_novelty([3, 4, 5, 6, 7])) / 2,
        id="two_users_mean_over_users",
    ),
]


def _novelty(config, rec_items):
    store = {
        "rec.items": _NumpyWrap(rec_items),
        "data.count_items": NOVELTY_COUNT_ITEMS,
        "data.num_items": BEYOND_UTILITY_NUM_ITEMS,
    }
    return metrics_dict["novelty"](config).calculate_metric(_DataObject(store))


@pytest.mark.parametrize(("rec_items", "expected"), NOVELTY_CASES)
def test_novelty(beyond_utility_config, rec_items, expected):
    """Novelty@k(u) = mean over the top-k recs of u of 1 - min-max normalized popularity, averaged."""
    result = _novelty(beyond_utility_config, rec_items)
    assert float(result["novelty@5"]) == pytest.approx(expected, abs=BEYOND_UTILITY_TOL)


def test_novelty_depends_on_k():
    result = _novelty(_beyond_utility_config([2, 5]), [[0, 1, 7, 8, 9]])
    got = {k: float(result[f"novelty@{k}"]) for k in (2, 5)}
    expected = {2: _mean_novelty([0, 1]), 5: _mean_novelty([0, 1, 7, 8, 9])}
    assert got == pytest.approx(expected, abs=BEYOND_UTILITY_TOL)


if __name__ == "__main__":
    unittest.main()
