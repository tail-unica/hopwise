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
config["topk"] = [K]


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

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

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


class TestExplainabilityLID(unittest.TestCase):
    def test_lid(self):
        name = "lid"
        Metric = metrics_dict[name](config)
        key = f"LID@{K}"
        dp = config["metric_decimal_place"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                user_total_paths = defaultdict(int)
                user_linking_ids = defaultdict(set)
                for user, _, _, path in case["paths"]:
                    linked_interaction_id = path[1][-1]
                    user_total_paths[user] += 1
                    user_linking_ids[user].add(linked_interaction_id)

                per_user_lid = []
                for user in user_total_paths:
                    n_paths = user_total_paths[user]
                    per_user_lid.append((len(user_linking_ids[user]) / n_paths) if n_paths else 0.0)

                expected = (sum(per_user_lid) / len(per_user_lid)) if per_user_lid else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


class TestExplainabilitySED(unittest.TestCase):
    def test_sed(self):
        name = "sed"
        Metric = metrics_dict[name](config)
        key = f"SED@{K}"
        dp = config["metric_decimal_place"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                user_total_paths = defaultdict(int)
                user_shared_entities = defaultdict(set)
                for user, _, _, path in case["paths"]:
                    shared_entity_id = path[-2][-1]
                    user_total_paths[user] += 1
                    user_shared_entities[user].add(shared_entity_id)

                per_user_sed = []
                for user in user_total_paths:
                    n_paths = user_total_paths[user]
                    per_user_sed.append((len(user_shared_entities[user]) / n_paths) if n_paths else 0.0)

                expected = (sum(per_user_sed) / len(per_user_sed)) if per_user_sed else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


class TestExplainabilityLITD(unittest.TestCase):
    def test_litd(self):
        name = "litd"
        Metric = metrics_dict[name](config)
        key = f"LITD@{K}"
        dp = config["metric_decimal_place"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                user_total_paths = defaultdict(int)
                user_linking_types = defaultdict(set)

                for user, _, _, path in case["paths"]:
                    linked_interaction_type = path[1][1]
                    user_total_paths[user] += 1
                    user_linking_types[user].add(linked_interaction_type)

                per_user_litd = []
                for user in user_total_paths:
                    n_paths = user_total_paths[user]
                    per_user_litd.append((len(user_linking_types[user]) / n_paths) if n_paths else 0.0)

                expected = (sum(per_user_litd) / len(per_user_litd)) if per_user_litd else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


class TestExplainabilitySETD(unittest.TestCase):
    def test_setd(self):
        name = "setd"
        Metric = metrics_dict[name](config)
        key = f"SETD@{K}"
        dp = config["metric_decimal_place"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"])
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                user_total_paths = defaultdict(int)
                user_shared_types = defaultdict(set)

                for user, _, _, path in case["paths"]:
                    shared_entity_type = path[-2][1]
                    user_total_paths[user] += 1
                    user_shared_types[user].add(shared_entity_type)

                per_user_setd = []
                for user in user_total_paths:
                    n_paths = user_total_paths[user]
                    per_user_setd.append((len(user_shared_types[user]) / n_paths) if n_paths else 0.0)

                expected = (sum(per_user_setd) / len(per_user_setd)) if per_user_setd else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


class TestExplainabilityPTD(unittest.TestCase):
    def test_ptd(self):
        name = "ptd"
        Metric = metrics_dict[name](config)
        key = f"PTD@{K}"
        dp = config["metric_decimal_place"]

        # For PTD we must provide: data.max_path_type
        # In the implementation, len(max_path_type) is used in the denominator. :contentReference[oaicite:1]{index=1}
        max_path_type = ["t0", "t1", "t2", "t3"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                # --- run metric ---
                dataobject = _DummyDataObject(case["paths"], max_path_type=max_path_type)
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                # --- expected (computed here, per case) ---
                # PTD.metric_info:
                #   path_type = path[-1][0]
                #   if path_type == "self_loop": path_type = path[-2][0]
                #   PTD_u = (#distinct path_type) / min(#paths_u, len(max_path_type))
                #   final = average over users
                user_total_paths = defaultdict(int)
                user_path_types = defaultdict(set)

                for user, _, _, path in case["paths"]:
                    path_type = path[-1][0]
                    if path_type == "self_loop":
                        path_type = path[-2][0]
                    user_total_paths[user] += 1
                    user_path_types[user].add(path_type)

                per_user_ptd = []
                for user in user_total_paths:
                    n_paths = user_total_paths[user]
                    denom = min(n_paths, len(max_path_type)) if n_paths else 1
                    per_user_ptd.append((len(user_path_types[user]) / denom) if n_paths else 0.0)

                expected = (sum(per_user_ptd) / len(per_user_ptd)) if per_user_ptd else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


if __name__ == "__main__":
    unittest.main()
