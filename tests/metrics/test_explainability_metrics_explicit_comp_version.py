# @Time    : 2026/01/24
# @Author  : Emanuele Caddeo

import os
import sys
import unittest
from collections import defaultdict

import numpy as np

sys.path.append(os.getcwd())

from hopwise.config import Config
from hopwise.evaluator.register import metrics_dict


parameters_dict = {
    "topk": [10],
    "metric_decimal_place": 4,
}

config = Config("BPR", "ml-1m", config_dict=parameters_dict)

K = 5
config["topk"] = [K]# Support functions (copy & paste)

def _extract_path_type_for_ptc(path):
    """
    Mirrors the original PTC implementation:
    - path_type is the first element of the last tuple in the path (path[-1][0])
    - if that is 'self_loop', use the previous one (path[-2][0])
    """
    path_type = path[-1][0]
    if path_type == "self_loop":
        path_type = path[-2][0]
    return path_type


def _expected_ptc_like_metric(case_paths, max_path_type):
    """
    Mirrors PTC.metric_info() + average in topk_result():
    - For each user:
        numerator = sum_r N(r) * (N(r) - 1)
        ptc_u = 1 - numerator / (N * (N - 1))   if N*(N-1) > 0 else 0
    - expected = mean(ptc_u over users)
    """
    user_simpson_index = {}  # user -> [N, {path_type: count}]
    max_type_set = set(max_path_type)

    for user, _, _, path in case_paths:
        if user not in user_simpson_index:
            user_simpson_index[user] = [0, {k: 0 for k in max_type_set}]

        path_type = _extract_path_type_for_ptc(path)
        if path_type not in user_simpson_index[user][1]:
            user_simpson_index[user][1][path_type] = 0

        user_simpson_index[user][1][path_type] += 1
        user_simpson_index[user][0] += 1

    per_user = []
    for user, (N, counts) in user_simpson_index.items():
        numerator = 0
        for _, n_i in counts.items():
            numerator += n_i * (n_i - 1)

        denom = N * (N - 1)
        if denom == 0:
            per_user.append(0.0)
        else:
            per_user.append(1.0 - (numerator / denom))

    return (sum(per_user) / len(per_user)) if per_user else 0.0



class _DummyDataObject:
    """
    Minimal dataobject stub for hopwise path explainability metrics.

    Current tests require:
      - rec.paths

    Future metrics may require:
      - LIR: data.timestamp, data.num_items
      - SEP: data.node_degree
      - PTD/PTC: data.max_path_type
      - PPT: data.max_path_length, data.rid2relation
    """

    def __init__(
        self,
        paths_value,
        timestamp_matrix=None,
        num_items=None,
        node_degree=None,
        max_path_type=None,
        max_path_length=None,
        rid2relation=None,
    ):
        self._store = {
            "rec.paths": paths_value,
            "data.timestamp": timestamp_matrix,
            "data.num_items": num_items,
            "data.node_degree": node_degree,
            "data.max_path_type": max_path_type,
            "data.max_path_length": max_path_length,
            "data.rid2relation": rid2relation,
        }

    def get(self, key):
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]


def _mk_path(user, rec_item, linking_item_id):
    """
    Build one path entry in the exact format expected by hopwise PathQuality metrics:
      (user, item, score, path)

    NOTE (based on hopwise/evaluator/metrics.py):
    - LID uses: linked_interaction_id = path[1][-1] (int)
    - SED uses: shared_entity_id = path[-2][-1]
    - LITD uses: linked_interaction_type = path[1][1]
    - SETD uses: shared_entity_type = path[-2][1]
    """
    path = [(0, "user", user), (1, "item", linking_item_id), (2, "entity", 999), (3, "item", rec_item)]
    return (user, rec_item, 0.0, path)


CASES = [
    {
        "name": "one_user_basic",
        "paths": [
            _mk_path(0, 20, 10),
            _mk_path(0, 21, 11),
            _mk_path(0, 22, 10),
            _mk_path(0, 23, 12),
        ],
    },
    {
        "name": "one_user_all_explainable_all_distinct",
        "paths": [
            _mk_path(0, 20, 10),
            _mk_path(0, 21, 11),
            _mk_path(0, 22, 12),
            _mk_path(0, 23, 13),
            _mk_path(0, 24, 14),
        ],
    },
    {
        "name": "one_user_sparse_all_same_linking",
        "paths": [
            _mk_path(0, 20, 10),
            _mk_path(0, 21, 10),
        ],
    },
    {
        "name": "two_users_mixed",
        "paths": [
            _mk_path(0, 20, 10),
            _mk_path(0, 21, 10),
            _mk_path(0, 22, 11),
            _mk_path(1, 30, 20),
            _mk_path(1, 31, 21),
            _mk_path(1, 32, 22),
        ],
    },
]


class TestExplainabilityFID(unittest.TestCase):
    def test_fidelity(self):
        name = "fidelity"
        Metric = metrics_dict[name](config)
        key = f"Fidelity@{K}"
        dp = config["metric_decimal_place"]

        self.assertEqual(
            [
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[0]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[1]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[2]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[3]["paths"]))[key]), dp),
            ],
            np.array(
                [
                    min(4 / K, 1.0),                         # one_user_basic: 4 paths -> 4/K
                    min(5 / K, 1.0),                         # one_user_all_explainable_all_distinct: 5 paths -> 5/K
                    min(2 / K, 1.0),                         # one_user_sparse_all_same_linking: 2 paths -> 2/K
                    min(((3 / K) + (3 / K)) / 2, 1.0),       # two_users_mixed: (3 paths user0 + 3 paths user1)/2
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilityLID(unittest.TestCase):
    def test_lid(self):
        name = "lid"
        Metric = metrics_dict[name](config)
        key = f"LID@{K}"
        dp = config["metric_decimal_place"]

        self.assertEqual(
            [
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[0]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[1]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[2]["paths"]))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[3]["paths"]))[key]), dp),
            ],
            np.array(
                [
                    # one_user_basic:
                    # linking ids = [10, 11, 10, 12] -> unique = 3, paths = 4 => 3/4
                    3 / 4,

                    # one_user_all_explainable_all_distinct:
                    # linking ids = [10, 11, 12, 13, 14] -> unique = 5, paths = 5 => 5/5
                    5 / 5,

                    # one_user_sparse_all_same_linking:
                    # linking ids = [10, 10] -> unique = 1, paths = 2 => 1/2
                    1 / 2,

                    # two_users_mixed:
                    # user0: [10,10,11] -> unique 2 / 3
                    # user1: [20,21,22] -> unique 3 / 3
                    # average = ((2/3) + (3/3)) / 2
                    ((2 / 3) + (3 / 3)) / 2,
                ]
            ).round(dp).tolist(),
        )


if __name__ == "__main__":
    unittest.main()