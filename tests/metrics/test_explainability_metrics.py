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

class _NumpyWrap:
    """Minimal wrapper exposing a .numpy() method."""
    def __init__(self, arr):
        self._arr = np.array(arr)

    def numpy(self):
        return self._arr


class _DummyRankingDataObject:
    """
    Minimal DataObject-like container exposing .get(key) for Serendipity.
    """
    def __init__(self, rec_items, num_items, num_users, count_items, history_index):
        self._store = {
            "rec.items": _NumpyWrap(rec_items),
            "data.num_items": num_items,
            "data.num_users": num_users,
            "data.count_items": count_items,
            "data.history_index": history_index,
        }

    def get(self, key):
        return self._store[key]


def _expected_serendipity_like_metric(rec_items, num_items, num_users, count_items, history_index, topk, dp):
    """
    Mirrors hopwise.evaluator.metrics.Serendipity.calculate_metric() + topk_result().
    """
    item_counter = np.zeros(num_items, dtype=int)
    for it, cnt in count_items.items():
        item_counter[int(it)] = int(cnt)

    pop_recs = np.tile(item_counter, (num_users, 1))
    pop_recs[tuple(history_index)] = 0
    pop_recs = pop_recs[1:]  # remove padding row

    max_k = max(topk)
    pop_topk = np.argsort(pop_recs, axis=-1)[:, ::-1][:, :max_k]

    rec_items = np.array(rec_items)
    topk_intersection = np.zeros_like(rec_items, dtype=bool)
    for u, (user_rec, user_pop_topk) in enumerate(zip(rec_items, pop_topk)):
        topk_intersection[u] = np.isin(user_rec, user_pop_topk)

    cumsum_pop = topk_intersection.cumsum(axis=1)
    denom = np.arange(1, topk_intersection.shape[1] + 1)
    ser_per_user_per_pos = 1.0 - (cumsum_pop / denom)

    avg_result = ser_per_user_per_pos.mean(axis=0)

    out = {}
    for k in topk:
        out[f"serendipity@{k}"] = round(float(avg_result[k - 1]), dp)
    return out


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
        # In the implementation, len(max_path_type) is used in the denominator.
        max_path_type = ["t0", "t1", "t2", "t3"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"], max_path_type=max_path_type)
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

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


class TestExplainabilityPTC(unittest.TestCase):
    def test_ptc(self):
        name = "ptc"
        Metric = metrics_dict[name](config)
        key = f"PTC@{K}"
        dp = config["metric_decimal_place"]

        # For PTC we must provide: data.max_path_type
        max_path_type = ["t0", "t1", "t2", "t3"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(case["paths"], max_path_type=max_path_type)
                out = Metric.calculate_metric(dataobject)
                self.assertIn(key, out)

                # Expected computed here (per case), mirroring the real metric:
                # - For each user, count occurrences per path_type.
                # - ptc = 1 - sum(n*(n-1)) / (N*(N-1)); if N*(N-1)==0 then ptc=0
                user_total_paths = defaultdict(int)
                user_path_type_counts = defaultdict(lambda: defaultdict(int))

                for user, _, _, path in case["paths"]:
                    path_type = path[-1][0]
                    if path_type == "self_loop":
                        path_type = path[-2][0]
                    user_total_paths[user] += 1
                    user_path_type_counts[user][path_type] += 1

                per_user_ptc = []
                for user in user_total_paths:
                    N = user_total_paths[user]
                    denom = N * (N - 1)
                    if denom == 0:
                        per_user_ptc.append(0.0)
                        continue
                    numerator = 0
                    for n in user_path_type_counts[user].values():
                        numerator += n * (n - 1)
                    per_user_ptc.append(1.0 - (numerator / denom))

                expected = (sum(per_user_ptc) / len(per_user_ptc)) if per_user_ptc else 0.0
                expected = round(expected, dp)

                self.assertEqual(float(out[key]), float(expected))


class TestExplainabilityPPT(unittest.TestCase):
    def test_ppt(self):
        name = "ppt"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]

        max_path_length = 4
        rid2relation = ["user", "item", "entity", "item"]

        # Build the same rid -> relation_name mapping used by the metric
        rid2r_name = {i: rel for i, rel in enumerate(rid2relation)}

        for case in CASES:
            with self.subTest(case=case["name"]):
                dataobject = _DummyDataObject(
                    case["paths"],
                    max_path_length=max_path_length,
                    rid2relation=rid2relation,
                )
                out = Metric.calculate_metric(dataobject)

                # The implementation returns the same averaged value for every k in Metric.topk
                # this is a property design choice. In the paper is not specified differently.
                # I assume that PPT does not depend on recommendation ranking.
                for k in Metric.topk:
                    key = f"PPT@{k}"
                    self.assertIn(key, out)

                unique_path_pattern = {}  # user -> [n_paths, set(pattern_strings)]
                for user, _, _, path in case["paths"]:
                    if user not in unique_path_pattern:
                        unique_path_pattern[user] = [0, set()]

                    path_pattern = [rid2r_name[path_tuple[0]] for path_tuple in path[1:]]
                    pattern_str = "_".join(path_pattern)

                    unique_path_pattern[user][0] += 1
                    unique_path_pattern[user][1].add(pattern_str)

                per_user_ppt = []
                for user, (n_paths, pattern_set) in unique_path_pattern.items():
                    n_path_patterns = len(pattern_set)
                    denom = min(n_paths, max_path_length) if max_path_length else 1
                    per_user_ppt.append(min(n_path_patterns / denom, 1.0))

                expected = round(sum(per_user_ppt) / len(per_user_ppt), dp) if per_user_ppt else round(0.0, dp)

                # Check that all PPT@k keys match the same expected value (implementation behavior)
                for k in Metric.topk:
                    key = f"PPT@{k}"
                    self.assertEqual(float(out[key]), float(expected))


class TestExplainabilityPTC(unittest.TestCase):
    def test_ptc(self):
        name = "ptc"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]

        for case in CASES:
            with self.subTest(case=case["name"]):
                # Build a max_path_type consistent with what PTC expects:
                # it's the universe of possible path types (relation types used as "explanation types").
                inferred_types = set()
                for _, _, _, path in case["paths"]:
                    inferred_types.add(_extract_path_type_for_ptc(path))
                max_path_type = sorted(inferred_types)

                dataobject = _DummyDataObject(
                    case["paths"],
                    max_path_type=max_path_type,
                )

                out = Metric.calculate_metric(dataobject)

                expected = _expected_ptc_like_metric(
                    case_paths=case["paths"],
                    max_path_type=max_path_type,
                )
                expected = round(expected, dp)

                # The original implementation copies the same averaged value to every @k key.
                for k in Metric.topk:
                    key = f"PTC@{k}"
                    self.assertIn(key, out)
                    self.assertEqual(float(out[key]), float(expected))

class TestBeyondUtilitySerendipity(unittest.TestCase):
    def test_serendipity(self):
        name = "serendipity"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]

        topk = Metric.topk
        max_k = max(topk)

        num_items = 50  # keep this > max_k and reasonably large
        # Popularity counts (higher = more popular)
        # Define counts for all items to avoid accidental ties/zeros dominating.
        count_items = {i: (num_items - i) for i in range(num_items)}  # item 0 most popular

        # Build rec.items with EXACTLY max_k items per user (required to avoid IndexError).
        # Use a deterministic pattern, all valid item ids in [0, num_items-1].
        rec_items = [
            [(3 + i) % num_items for i in range(max_k)],   # user0
            [(10 + i) % num_items for i in range(max_k)],  # user1
        ]

        # data.num_users includes the padding row 0
        num_users = len(rec_items) + 1

        # Mark some history items as seen for each user (rows 1..num_users-1).
        # Keep indices in-bounds.
        history_index = np.array(
            [
                [1, 1, 2, 2],  # row indices
                [0, 1, 2, 3],  # item indices
            ],
            dtype=int,
        )

        dataobject = _DummyRankingDataObject(
            rec_items=rec_items,
            num_items=num_items,
            num_users=num_users,
            count_items=count_items,
            history_index=history_index,
        )

        out = Metric.calculate_metric(dataobject)

        expected = _expected_serendipity_like_metric(
            rec_items=rec_items,
            num_items=num_items,
            num_users=num_users,
            count_items=count_items,
            history_index=history_index,
            topk=topk,
            dp=dp,
        )

        for k in topk:
            key = f"serendipity@{k}"
            self.assertIn(key, out)
            self.assertEqual(float(out[key]), float(expected[key]))

            
if __name__ == "__main__":
    unittest.main()