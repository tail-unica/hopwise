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

K = 5  # we want @5 behavior in this unit test file
config["topk"] = [K]


class _DummyDataObject:
    """
    Minimal dataobject stub for hopwise metrics.

    Current tests (Fidelity/LID) only require:
      - rec.paths

    Future path explainability metrics require additional keys (as in hopwise/evaluator/metrics.py):
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

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                # Expected computed here (per case)
                user_items = defaultdict(set)
                for user, item, _, _ in case["paths"]:
                    user_items[user].add(item)

                per_user_counts = [len(items) for items in user_items.values()]
                expected = (
                    min(sum((c / K) for c in per_user_counts) / len(per_user_counts), 1.0)
                    if per_user_counts
                    else 0.0
                )
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


if __name__ == "__main__":
    unittest.main()
