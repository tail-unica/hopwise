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

    def test_serendipity(self):
        name = "serendipity"

        # Use a small k to keep the test compact and deterministic.
        local_parameters = {"topk": [5], "metric_decimal_place": 4}
        local_config = Config("BPR", "ml-1m", config_dict=local_parameters)

        Metric = metrics_dict[name](local_config)
        k = local_config["topk"][0]
        key = f"serendipity@{k}"
        dp = local_config["metric_decimal_place"]

        class _RecWrap:
            def __init__(self, arr):
                self._arr = np.array(arr, dtype=int)

            def numpy(self):
                return self._arr

        class _MinimalDataObject:
            def __init__(self, rec_items, count_items, num_items, num_users, history_index):
                self._store = {
                    "rec.items": _RecWrap(rec_items),
                    "data.count_items": count_items,
                    "data.num_items": num_items,
                    "data.num_users": num_users,
                    "data.history_index": history_index,
                }

            def get(self, key):
                return self._store[key]

        num_items = 10
        # item 0 most popular, item 9 least popular
        count_items = {i: (num_items - i) for i in range(num_items)}

        cases = [
            {
                "name": "all_recs_are_popular_for_both_users",
                # Recommend top popular items -> intersection@k = k -> serendipity = 0
                "rec_items": [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                ],
                "history_index": np.empty((2, 0), dtype=int),
                "expected": 0.0,
            },
            {
                "name": "no_recs_are_popular_for_both_users",
                # Recommend least popular items -> intersection@k = 0 -> serendipity = 1
                "rec_items": [
                    [9, 8, 7, 6, 5],
                    [9, 8, 7, 6, 5],
                ],
                "history_index": np.empty((2, 0), dtype=int),
                "expected": 1.0,
            },
            {
                "name": "mixed_popularity_intersection_three_out_of_five",
                # 3 popular items in user's popularity-topk -> serendipity = 1 - 3/5 = 0.4
                "rec_items": [
                    [0, 1, 2, 9, 8],
                    [0, 1, 2, 9, 8],
                ],
                "history_index": np.empty((2, 0), dtype=int),
                "expected": 1 - (3 / 5),
            },
            {
                "name": "history_removal_increases_serendipity",
                # Remove {0,1} from popularity list for both users via history.
                # history_index must be shaped as (2, N): [user_ids; item_ids]
                # After removal, only item 2 remains in the popularity-topk intersection -> 1 - 1/5 = 0.8
                "rec_items": [
                    [0, 1, 2, 9, 8],
                    [0, 1, 2, 9, 8],
                ],
                "history_index": np.array(
                    [
                        [1, 1, 2, 2],  # user ids (1-indexed due to padding)
                        [0, 1, 0, 1],  # item ids
                    ],
                    dtype=int,
                ),
                "expected": 1 - (1 / 5),
            },
            {
                "name": "two_users_opposite_lists_average",
                # user0: all popular -> 0
                # user1: none popular -> 1
                # mean -> 0.5
                "rec_items": [
                    [0, 1, 2, 3, 4],
                    [9, 8, 7, 6, 5],
                ],
                "history_index": np.empty((2, 0), dtype=int),
                "expected": 0.5,
            },
        ]

        got = []
        expected = []

        for c in cases:
            rec_items = c["rec_items"]
            num_users = len(rec_items) + 1  # padding row 0

            dataobject = _MinimalDataObject(
                rec_items=rec_items,
                count_items=count_items,
                num_items=num_items,
                num_users=num_users,
                history_index=c["history_index"],
            )

            metric_value = float(Metric.calculate_metric(dataobject)[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)

    def test_novelty(self):
        name = "novelty"

        local_parameters = {"topk": [5], "metric_decimal_place": 4}
        local_config = Config("BPR", "ml-1m", config_dict=local_parameters)

        Metric = metrics_dict[name](local_config)
        k = local_config["topk"][0]
        key = f"novelty@{k}"
        dp = local_config["metric_decimal_place"]

        class _RecWrap:
            def __init__(self, arr):
                self._arr = np.array(arr, dtype=int)

            def numpy(self):
                return self._arr

        class _MinimalDataObject:
            def __init__(self, rec_items, count_items, num_items):
                self._store = {
                    "rec.items": _RecWrap(rec_items),
                    "data.count_items": count_items,
                    "data.num_items": num_items,
                }

            def get(self, key):
                return self._store[key]

        num_items = 10
        # item 0 -> count=1 (least popular), item 9 -> count=10 (most popular)
        count_items = {i: (i + 1) for i in range(num_items)}

        # With min_pop=1, max_pop=10:
        # normalized_pop(i) = (count(i)-1)/9 = i/9
        # novelty(i) = 1 - i/9
        def nov(i: int) -> float:
            return 1.0 - (i / 9.0)

        cases = [
            {
                "name": "all_low_pop_items_high_novelty",
                "rec_items": [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                ],
                "expected": sum(nov(i) for i in [0, 1, 2, 3, 4]) / 5,
            },
            {
                "name": "all_high_pop_items_low_novelty",
                "rec_items": [
                    [5, 6, 7, 8, 9],
                    [5, 6, 7, 8, 9],
                ],
                "expected": sum(nov(i) for i in [5, 6, 7, 8, 9]) / 5,
            },
            {
                "name": "two_users_different_lists_mean_over_users",
                "rec_items": [
                    [0, 1, 2, 3, 4],
                    [3, 4, 5, 6, 7],
                ],
                "expected": (
                    (sum(nov(i) for i in [0, 1, 2, 3, 4]) / 5)
                    + (sum(nov(i) for i in [3, 4, 5, 6, 7]) / 5)
                )
                / 2,
            },
            {
                "name": "uniform_mid_pop_items",
                "rec_items": [
                    [2, 3, 4, 5, 6],
                    [2, 3, 4, 5, 6],
                ],
                "expected": sum(nov(i) for i in [2, 3, 4, 5, 6]) / 5,
            },
        ]

        got = []
        expected = []

        for c in cases:
            dataobject = _MinimalDataObject(
                rec_items=c["rec_items"],
                count_items=count_items,
                num_items=num_items,
            )
            metric_value = float(Metric.calculate_metric(dataobject)[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
