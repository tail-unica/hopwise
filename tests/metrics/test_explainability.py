# @Time    : 2026/01/26
# @Author  : Emanuele Caddeo

import os
import sys
import unittest

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


# -------------------------------------------------------------------------
# Support functions (copy & paste)
# -------------------------------------------------------------------------


class _NumpyWrap:
    def __init__(self, arr):
        self._arr = np.array(arr)

    def numpy(self):
        return self._arr


class _DummyDataObject:
    """
    Minimal dataobject stub for hopwise metrics.

    Keep fields aligned with what metrics access via `dataobject.get(key)`.
    """

    def __init__(
        self,
        paths_value=None,
        timestamp_matrix=None,
        num_items=None,
        node_degree=None,
        max_path_type=None,
        max_path_length=None,
        rid2relation=None,
        # beyond-utility placeholders
        rec_items=None,
        count_items=None,
        num_users=None,
        history_index=None,
    ):
        self._store = {
            "rec.paths": paths_value,
            "data.timestamp": timestamp_matrix,
            "data.num_items": num_items,
            "data.node_degree": node_degree,
            "data.max_path_type": max_path_type,
            "data.max_path_length": max_path_length,
            "data.rid2relation": rid2relation,
            # beyond-utility
            "rec.items": _NumpyWrap(rec_items) if rec_items is not None else None,
            "data.count_items": count_items,
            "data.num_users": num_users,
            "data.history_index": history_index,
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


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestExplainabilityFID(unittest.TestCase):
    def test_fid(self):
        """
        Fidelity@K: percentage of recommended items that can be explained.
        In hopwise implementation, it is effectively min(n_paths/K, 1.0) averaged across users.
        """
        name = "fidelity"
        Metric = metrics_dict[name](config)
        key = f"Fidelity@{K}"
        dp = config["metric_decimal_place"]

        # Define cases *inside* the test (so each test can tailor edge cases).
        cases = [
            {
                "name": "one_user_basic",
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 11),
                    _mk_path(0, 22, 10),
                    _mk_path(0, 23, 12),
                ],
                "expected": min(4 / K, 1.0),
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
                "expected": min(5 / K, 1.0),
            },
            {
                "name": "one_user_sparse_all_same_linking",
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 10),
                ],
                "expected": min(2 / K, 1.0),
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
                "expected": min(((3 / K) + (3 / K)) / 2, 1.0),
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(Metric.calculate_metric(_DummyDataObject(paths_value=c["paths"]))[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilityLID(unittest.TestCase):
    def test_lid(self):
        """
        LID (Linked Interaction Diversity):

        For each user:
          LID(u) = (#unique linked_interaction_id) / (#paths for that user)

        Then averaged across users.

        Edge-case focus:
        - many paths with same linked_interaction_id (should be low, not 1.0)
        - all paths with distinct linked_interaction_id (should be 1.0)
        - mixed duplicates (fractional value)
        - multi-user mean (must average per-user, not aggregate globally)
        """
        name = "lid"
        Metric = metrics_dict[name](config)
        key = f"LID@{K}"
        dp = config["metric_decimal_place"]

        cases = [
            {
                "name": "one_user_all_same_linked_interaction",
                # n_paths=5, unique_linked_interaction=1 => 1/5
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 10),
                    _mk_path(0, 22, 10),
                    _mk_path(0, 23, 10),
                    _mk_path(0, 24, 10),
                ],
                "expected": 1 / 5,
            },
            {
                "name": "one_user_all_distinct_linked_interaction",
                # n_paths=4, unique=4 => 1.0
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 11),
                    _mk_path(0, 22, 12),
                    _mk_path(0, 23, 13),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_duplicates",
                # linked_interaction ids: {10,11,12} with counts 2,2,1 => n_paths=5, unique=3 => 3/5
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 10),
                    _mk_path(0, 22, 11),
                    _mk_path(0, 23, 11),
                    _mk_path(0, 24, 12),
                ],
                "expected": 3 / 5,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_ratios",
                # user0: n_paths=5 unique=1 => 1/5
                # user1: n_paths=4 unique=4 => 1
                # mean => (1/5 + 1)/2 = 0.6
                "paths": [
                    _mk_path(0, 20, 10),
                    _mk_path(0, 21, 10),
                    _mk_path(0, 22, 10),
                    _mk_path(0, 23, 10),
                    _mk_path(0, 24, 10),
                    _mk_path(1, 30, 20),
                    _mk_path(1, 31, 21),
                    _mk_path(1, 32, 22),
                    _mk_path(1, 33, 23),
                ],
                "expected": ((1 / 5) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(Metric.calculate_metric(_DummyDataObject(paths_value=c["paths"]))[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilitySED(unittest.TestCase):
    def test_sed(self):
        """
        SED (Shared Entity Diversity):

        For each user:
          SED(u) = (#unique shared_entity_id) / (#paths for that user)

        Then averaged across users.

        Edge-case focus (formula failures):
        - all paths share the same entity -> low value (1/n_paths), not 1.0
        - every path has a different entity -> must be exactly 1.0
        - mixed duplicates -> fractional value (unique/n_paths)
        - multi-user averaging -> must average per-user ratios (no global aggregation)
        """
        name = "sed"
        Metric = metrics_dict[name](config)
        key = f"SED@{K}"
        dp = config["metric_decimal_place"]

        def _mk_path_with_shared_entity(user, rec_item, linking_item_id, shared_entity_id):
            """
            Build a path where the shared entity id is controllable.

            The metric reads:
              shared_entity_id = path[-2][-1]
            so we place it at the penultimate triple.
            """
            path = [
                (0, "user", user),
                (1, "item", linking_item_id),
                (2, "entity", shared_entity_id),  # <- shared entity (penultimate node)
                (3, "item", rec_item),
            ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "one_user_all_same_shared_entity",
                # n_paths=5, unique_shared_entity=1 => 1/5
                "paths": [
                    _mk_path_with_shared_entity(0, 20, 10, 999),
                    _mk_path_with_shared_entity(0, 21, 11, 999),
                    _mk_path_with_shared_entity(0, 22, 12, 999),
                    _mk_path_with_shared_entity(0, 23, 13, 999),
                    _mk_path_with_shared_entity(0, 24, 14, 999),
                ],
                "expected": 1 / 5,
            },
            {
                "name": "one_user_all_distinct_shared_entity",
                # n_paths=4, unique=4 => 1.0
                "paths": [
                    _mk_path_with_shared_entity(0, 20, 10, 900),
                    _mk_path_with_shared_entity(0, 21, 11, 901),
                    _mk_path_with_shared_entity(0, 22, 12, 902),
                    _mk_path_with_shared_entity(0, 23, 13, 903),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_duplicates_shared_entity",
                # shared entities: {900,901,902} with counts 2,2,1 => unique=3, n_paths=5 => 3/5
                "paths": [
                    _mk_path_with_shared_entity(0, 20, 10, 900),
                    _mk_path_with_shared_entity(0, 21, 11, 900),
                    _mk_path_with_shared_entity(0, 22, 12, 901),
                    _mk_path_with_shared_entity(0, 23, 13, 901),
                    _mk_path_with_shared_entity(0, 24, 14, 902),
                ],
                "expected": 3 / 5,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_ratios",
                # user0: n_paths=3 unique=1 => 1/3
                # user1: n_paths=2 unique=2 => 1
                # mean => (1/3 + 1)/2 = 0.6667
                "paths": [
                    _mk_path_with_shared_entity(0, 20, 10, 999),
                    _mk_path_with_shared_entity(0, 21, 11, 999),
                    _mk_path_with_shared_entity(0, 22, 12, 999),
                    _mk_path_with_shared_entity(1, 30, 20, 800),
                    _mk_path_with_shared_entity(1, 31, 21, 801),
                ],
                "expected": ((1 / 3) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(Metric.calculate_metric(_DummyDataObject(paths_value=c["paths"]))[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilityLITD(unittest.TestCase):
    def test_litd(self):
        """
        LITD (Linked Interaction Type Diversity):

        For each user:
          LITD(u) = (#unique linked_interaction_type) / (#paths for that user)

        Then averaged across users.

        Edge-case focus (formula failures):
        - many paths with different linked IDs but same type -> should be low (1/n_paths)
        - every path has a different type -> must be exactly 1.0
        - mixed duplicates -> fractional value (unique/n_paths)
        - multi-user mean -> must average per-user ratios (no global aggregation)
        """
        name = "litd"
        Metric = metrics_dict[name](config)
        key = f"LITD@{K}"
        dp = config["metric_decimal_place"]

        def _mk_path_with_link_type(user, rec_item, linking_type, linking_item_id, shared_entity_id=999):
            """
            Build a path where the linked interaction TYPE is controllable.

            The metric reads:
              linked_interaction_type = path[1][1]
            so we set the second element of the 2nd tuple (index 1).
            """
            path = [
                (0, "user", user),
                (1, linking_type, linking_item_id),  # <- linked interaction type here
                (2, "entity", shared_entity_id),
                (3, "item", rec_item),
            ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "one_user_many_ids_same_type_counts_type_not_id",
                # n_paths=5, linked types all 'item' => unique_types=1 => 1/5
                # If someone mistakenly counts linked IDs, they'd get 5/5 = 1.0 (WRONG).
                "paths": [
                    _mk_path_with_link_type(0, 20, "item", 10),
                    _mk_path_with_link_type(0, 21, "item", 11),
                    _mk_path_with_link_type(0, 22, "item", 12),
                    _mk_path_with_link_type(0, 23, "item", 13),
                    _mk_path_with_link_type(0, 24, "item", 14),
                ],
                "expected": 1 / 5,
            },
            {
                "name": "one_user_all_distinct_types",
                # n_paths=4, types all distinct => unique=4 => 1.0
                "paths": [
                    _mk_path_with_link_type(0, 20, "item", 10),
                    _mk_path_with_link_type(0, 21, "entity", 11),
                    _mk_path_with_link_type(0, 22, "user", 12),
                    _mk_path_with_link_type(0, 23, "brand", 13),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_duplicate_types",
                # types: item,item,entity,entity,brand => unique=3, n_paths=5 => 3/5
                "paths": [
                    _mk_path_with_link_type(0, 20, "item", 10),
                    _mk_path_with_link_type(0, 21, "item", 11),
                    _mk_path_with_link_type(0, 22, "entity", 12),
                    _mk_path_with_link_type(0, 23, "entity", 13),
                    _mk_path_with_link_type(0, 24, "brand", 14),
                ],
                "expected": 3 / 5,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_ratios",
                # user0: n_paths=3 types all 'item' => 1/3
                # user1: n_paths=2 types distinct => 2/2=1
                # mean => (1/3 + 1)/2 = 0.6667
                "paths": [
                    _mk_path_with_link_type(0, 20, "item", 10),
                    _mk_path_with_link_type(0, 21, "item", 11),
                    _mk_path_with_link_type(0, 22, "item", 12),
                    _mk_path_with_link_type(1, 30, "item", 20),
                    _mk_path_with_link_type(1, 31, "entity", 21),
                ],
                "expected": ((1 / 3) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(Metric.calculate_metric(_DummyDataObject(paths_value=c["paths"]))[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilitySETD(unittest.TestCase):
    def test_setd(self):
        """
        SETD (Shared Entity Type Diversity):

        For each user:
          SETD(u) = (#unique shared_entity_type) / (#paths for that user)

        Then averaged across users.

        Edge-case focus (formula failures):
        - many paths with different shared entity IDs but same TYPE -> should be low (1/n_paths)
          (catches bugs counting IDs instead of types)
        - distinct shared entity TYPES -> must be exactly 1.0
        - mixed duplicate types -> fractional value (unique/n_paths)
        - indexing bug guard: ensure shared entity type is read from path[-2][1], not path[-1][1]
        - multi-user mean -> must average per-user ratios (no global aggregation)
        """
        name = "setd"
        Metric = metrics_dict[name](config)
        key = f"SETD@{K}"
        dp = config["metric_decimal_place"]

        def _mk_path_with_shared_entity_type(
            user,
            rec_item,
            linking_item_id,
            shared_entity_type,
            shared_entity_id,
        ):
            """
            Build a path where the shared entity TYPE is controllable.

            The metric reads:
              shared_entity_type = path[-2][1]
            so we set the second element of the penultimate tuple.

            NOTE: We keep the last node type fixed to 'item' so if someone mistakenly reads path[-1][1],
            they would see only 'item' and the diversity would collapse (test should fail).
            """
            path = [
                (0, "user", user),
                (1, "item", linking_item_id),
                (2, shared_entity_type, shared_entity_id),  # <- shared entity type here (penultimate)
                (3, "item", rec_item),                      # <- final node type always 'item'
            ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "one_user_many_ids_same_shared_entity_type_counts_type_not_id",
                # n_paths=5, shared entity type always 'entity' but IDs differ => unique_types=1 => 1/5
                # If someone mistakenly counts IDs, they'd get 5/5 = 1.0 (WRONG).
                # If someone mistakenly reads path[-1][1] they'd get only 'item' => 1/5 as well,
                # so we also include a dedicated indexing-guard case below where types vary.
                "paths": [
                    _mk_path_with_shared_entity_type(0, 20, 10, "entity", 900),
                    _mk_path_with_shared_entity_type(0, 21, 11, "entity", 901),
                    _mk_path_with_shared_entity_type(0, 22, 12, "entity", 902),
                    _mk_path_with_shared_entity_type(0, 23, 13, "entity", 903),
                    _mk_path_with_shared_entity_type(0, 24, 14, "entity", 904),
                ],
                "expected": 1 / 5,
            },
            {
                "name": "one_user_all_distinct_shared_entity_types_indexing_guard",
                # n_paths=4, shared entity TYPES all distinct => unique_types=4 => 1.0
                # If someone mistakenly reads path[-1][1], they'd see only 'item' => 1/4 (WRONG).
                "paths": [
                    _mk_path_with_shared_entity_type(0, 20, 10, "entity", 900),
                    _mk_path_with_shared_entity_type(0, 21, 11, "brand", 901),
                    _mk_path_with_shared_entity_type(0, 22, 12, "genre", 902),
                    _mk_path_with_shared_entity_type(0, 23, 13, "tag", 903),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_duplicate_shared_entity_types",
                # types: entity,entity,brand,brand,genre => unique=3, n_paths=5 => 3/5
                "paths": [
                    _mk_path_with_shared_entity_type(0, 20, 10, "entity", 900),
                    _mk_path_with_shared_entity_type(0, 21, 11, "entity", 901),
                    _mk_path_with_shared_entity_type(0, 22, 12, "brand", 902),
                    _mk_path_with_shared_entity_type(0, 23, 13, "brand", 903),
                    _mk_path_with_shared_entity_type(0, 24, 14, "genre", 904),
                ],
                "expected": 3 / 5,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_ratios",
                # user0: n_paths=3 types all 'entity' => 1/3
                # user1: n_paths=2 types distinct => 2/2=1
                # mean => (1/3 + 1)/2 = 0.6667
                "paths": [
                    _mk_path_with_shared_entity_type(0, 20, 10, "entity", 900),
                    _mk_path_with_shared_entity_type(0, 21, 11, "entity", 901),
                    _mk_path_with_shared_entity_type(0, 22, 12, "entity", 902),
                    _mk_path_with_shared_entity_type(1, 30, 20, "entity", 800),
                    _mk_path_with_shared_entity_type(1, 31, 21, "brand", 801),
                ],
                "expected": ((1 / 3) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(Metric.calculate_metric(_DummyDataObject(paths_value=c["paths"]))[key])
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilityPPT(unittest.TestCase):
    def test_ppt(self):
        """
        PPT (Path Pattern Type):

        For each user:
          - Build a pattern as relation-names of path[1:] using path_tuple[0] -> rid2relation[rid]
          - ppt(u) = min( n_unique_patterns / min(n_paths, max_path_length), 1.0 )
        Then averaged across users.

        Edge-case focus (formula failures):
        - pattern identity must depend ONLY on relation-ids, not entity/item IDs
        - denominator must be min(n_paths, max_path_length), not always max_path_length
        - saturation when n_paths > max_path_length and all patterns are distinct -> must cap to 1.0
        - multi-user mean must average per-user values
        """
        name = "ppt"
        Metric = metrics_dict[name](config)
        key = f"PPT@{K}"
        dp = config["metric_decimal_place"]

        # Keep rid2relation long enough to cover all rids used in our paths.
        rid2relation = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]

        def _mk_path_with_rel_ids(user, rec_item, linking_item_id, rel_ids):
            """
            Build a path where the relation-id sequence is controllable.

            IMPORTANT: PPT uses:
              path_pattern = [rid2r_name[path_tuple[0]] for path_tuple in path[1:]]
            so only the first element of each tuple in path[1:] matters.

            rel_ids is a list of 4 ints for the 4 tuples:
              [rid_user, rid_link, rid_shared_entity, rid_target_item]
            but PPT ignores rel_ids[0] because it starts from path[1:].
            """
            assert len(rel_ids) == 4

            path = [
                (rel_ids[0], "user", user),
                (rel_ids[1], "item", linking_item_id),
                (rel_ids[2], "entity", 999),
                (rel_ids[3], "item", rec_item),
            ]
            return (user, rec_item, 0.0, path)

        max_path_length = 4

        cases = [
            {
                "name": "same_relation_pattern_different_item_ids_must_dedup_to_one_pattern",
                # All paths share the SAME rid-sequence in path[1:], but item/link IDs change.
                # Unique patterns = 1
                # n_paths = 4, denom = min(4,4)=4 -> 1/4
                # Catches bug: if someone includes entity/item IDs in the pattern, they'd get 4/4 = 1.0 (WRONG).
                "paths": [
                    _mk_path_with_rel_ids(0, 20, 10, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 21, 11, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 22, 12, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 23, 13, [0, 1, 2, 3]),
                ],
                "expected": 1 / 4,
            },
            {
                "name": "n_paths_smaller_than_max_length_denominator_must_be_n_paths",
                # n_paths = 2, max_path_length = 4, and patterns are DISTINCT -> unique=2
                # denom must be min(2,4)=2 -> 2/2=1.0
                # Catches bug: if denom incorrectly uses max_path_length (4), you'd get 0.5 (WRONG).
                "paths": [
                    _mk_path_with_rel_ids(0, 20, 10, [0, 1, 2, 3]),  # pattern: r1_r2_r3
                    _mk_path_with_rel_ids(0, 21, 11, [0, 4, 2, 3]),  # pattern: r4_r2_r3 (different)
                ],
                "expected": 1.0,
            },
            {
                "name": "saturation_when_more_paths_than_max_length_all_patterns_distinct",
                # n_paths = 6, max_path_length = 4, all patterns DISTINCT -> unique=6
                # denom = min(6,4)=4 -> 6/4=1.5, cap -> 1.0
                "paths": [
                    _mk_path_with_rel_ids(0, 20, 10, [0, 1, 2, 3]),  # r1_r2_r3
                    _mk_path_with_rel_ids(0, 21, 11, [0, 4, 2, 3]),  # r4_r2_r3
                    _mk_path_with_rel_ids(0, 22, 12, [0, 1, 5, 3]),  # r1_r5_r3
                    _mk_path_with_rel_ids(0, 23, 13, [0, 1, 2, 6]),  # r1_r2_r6
                    _mk_path_with_rel_ids(0, 24, 14, [0, 4, 5, 3]),  # r4_r5_r3
                    _mk_path_with_rel_ids(0, 25, 15, [0, 4, 2, 6]),  # r4_r2_r6
                ],
                "expected": 1.0,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_values",
                # user0: 4 paths all same pattern -> 1/min(4,4)=1/4
                # user1: 2 paths distinct patterns with n_paths<max -> 2/min(2,4)=1
                # mean -> (1/4 + 1)/2 = 0.625
                "paths": [
                    _mk_path_with_rel_ids(0, 20, 10, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 21, 11, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 22, 12, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(0, 23, 13, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(1, 30, 20, [0, 1, 2, 3]),
                    _mk_path_with_rel_ids(1, 31, 21, [0, 4, 2, 3]),
                ],
                "expected": ((1 / 4) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        for c in cases:
            metric_value = float(
                Metric.calculate_metric(
                    _DummyDataObject(
                        paths_value=c["paths"],
                        max_path_length=max_path_length,
                        rid2relation=rid2relation,
                    )
                )[key]
            )
            got.append(round(metric_value, dp))
            expected.append(round(float(c["expected"]), dp))

        self.assertEqual(got, expected)


class TestExplainabilityPTC(unittest.TestCase):
    def test_ptc(self):
        """
        PTC (Path Type Concentration):

        For each user:
          PTC(u) = 1 - ( sum_t n_t (n_t - 1) ) / ( N (N - 1) )

        Then averaged across users.

        This test focuses on:
        - correctness of the Simpson-like concentration formula
        - explicit verification that PTC is invariant w.r.t. k
          (PTC@5 == PTC@10 == ... in the current implementation)
        """
        name = "ptc"
        Metric = metrics_dict[name](config)

        # We explicitly test multiple k values
        ks = config["topk"]
        dp = config["metric_decimal_place"]

        # PTC needs max_path_type to build internal counters
        max_path_type = [0, 1, 2, 3]

        def _mk_path_with_type(user, rec_item, linking_item_id, path_type):
            """
            Build a path with a controllable path_type.

            PTC extracts the path type from path[-1][0],
            unless it is 'self_loop' (not needed for this test).
            """
            path = [
                (0, "user", user),
                (1, "item", linking_item_id),
                (2, "entity", 999),
                (path_type, "item", rec_item),  # <- path type
            ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "one_user_all_same_type",
                # N=4, counts={0:4}
                # PTC = 1 - (4*3)/(4*3) = 0
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0),
                    _mk_path_with_type(0, 22, 12, 0),
                    _mk_path_with_type(0, 23, 13, 0),
                ],
                "expected": 0.0,
            },
            {
                "name": "one_user_all_distinct_types",
                # N=4, counts={0:1,1:1,2:1,3:1}
                # PTC = 1 - 0/(4*3) = 1
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 1),
                    _mk_path_with_type(0, 22, 12, 2),
                    _mk_path_with_type(0, 23, 13, 3),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_distribution",
                # N=5, counts={0:3,1:2}
                # sum n_t(n_t-1) = 3*2 + 2*1 = 8
                # denom = 5*4 = 20
                # PTC = 1 - 8/20 = 0.6
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0),
                    _mk_path_with_type(0, 22, 12, 0),
                    _mk_path_with_type(0, 23, 13, 1),
                    _mk_path_with_type(0, 24, 14, 1),
                ],
                "expected": 0.6,
            },
            {
                "name": "two_users_mean_of_user_values",
                # user0: all same type, N=4 -> 0.0
                # user1: mixed 3 vs 2, N=5 -> 0.6
                # mean = 0.3
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0),
                    _mk_path_with_type(0, 22, 12, 0),
                    _mk_path_with_type(0, 23, 13, 0),
                    _mk_path_with_type(1, 30, 20, 0),
                    _mk_path_with_type(1, 31, 21, 0),
                    _mk_path_with_type(1, 32, 22, 0),
                    _mk_path_with_type(1, 33, 23, 1),
                    _mk_path_with_type(1, 34, 24, 1),
                ],
                "expected": 0.3,
            },
        ]

        for c in cases:
            metric_dict = Metric.calculate_metric(
                _DummyDataObject(
                    paths_value=c["paths"],
                    max_path_type=max_path_type,
                )
            )
            values = []
            for k in ks:
                key = f"PTC@{k}"
                values.append(round(float(metric_dict[key]), dp))

            expected_value = round(float(c["expected"]), dp)
            self.assertTrue(all(v == expected_value for v in values))


class TestExplainabilityPTD(unittest.TestCase):
    def test_ptd(self):
        """
        PTD (Path Type Diversity):

        For each user:
          path_type = path[-1][0], unless it is "self_loop", then path_type = path[-2][0]
          PTD(u) = (#unique path_type) / min(n_paths_u, len(max_path_type))
        Then averaged across users.

        Edge-case focus (formula failures):
        - single type repeated -> 1/min(n_paths, |T|)
        - all distinct types with n_paths <= |T| -> 1.0
        - more unique types than |T| (with n_paths >= |T|) -> PTD can be > 1.0 in current implementation
        - self_loop indexing must read type from path[-2][0]
        - multi-user mean must average per-user values
        """
        name = "ptd"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]

        # PTD needs max_path_type (only its LENGTH matters for the denominator)
        max_path_type = [0, 1, 2, 3]  # len = 4
        T = len(max_path_type)

        def _mk_path_with_type(user, rec_item, linking_item_id, path_type, use_self_loop=False):
            """
            Build a path with a controllable extracted path_type.

            PTD extracts:
              path_type = path[-1][0]
              if path_type == "self_loop": path_type = path[-2][0]
            """
            if not use_self_loop:
                path = [
                    (0, "user", user),
                    (1, "item", linking_item_id),
                    (2, "entity", 999),
                    (path_type, "item", rec_item),  # <- extracted type
                ]
            else:
                path = [
                    (0, "user", user),
                    (1, "item", linking_item_id),
                    (path_type, "entity", 999),      # <- extracted from here when self_loop
                    ("self_loop", "item", rec_item), # <- triggers special handling
                ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "one_user_all_same_type_n_paths_gt_T",
                # n_paths=5, unique_types=1, denom=min(5,T=4)=4 => 1/4
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0),
                    _mk_path_with_type(0, 22, 12, 0),
                    _mk_path_with_type(0, 23, 13, 0),
                    _mk_path_with_type(0, 24, 14, 0),
                ],
                "expected": 1 / 4,
            },
            {
                "name": "one_user_all_distinct_types_n_paths_le_T",
                # n_paths=4, unique_types=4, denom=min(4,4)=4 => 1.0
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 1),
                    _mk_path_with_type(0, 22, 12, 2),
                    _mk_path_with_type(0, 23, 13, 3),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_more_unique_types_than_T_can_exceed_one",
                # n_paths=5, unique_types=5, denom=min(5,4)=4 => 5/4 = 1.25 (no cap in current impl)
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 1),
                    _mk_path_with_type(0, 22, 12, 2),
                    _mk_path_with_type(0, 23, 13, 3),
                    _mk_path_with_type(0, 24, 14, 4),
                ],
                "expected": 5 / 4,
            },
            {
                "name": "self_loop_type_must_be_taken_from_penultimate",
                # n_paths=4, extracted types should be {0,0,1,1} => unique=2, denom=min(4,4)=4 => 2/4 = 0.5
                # If indexing is wrong and it uses 'self_loop' as type, it would change the set and fail.
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0, use_self_loop=True),
                    _mk_path_with_type(0, 22, 12, 1),
                    _mk_path_with_type(0, 23, 13, 1, use_self_loop=True),
                ],
                "expected": 2 / 4,
            },
            {
                "name": "two_users_mean_must_be_average_of_user_values",
                # user0: all same type, n_paths=5 -> 1/4 = 0.25
                # user1: all distinct (4) n_paths=4 -> 1.0
                # mean -> 0.625
                "paths": [
                    _mk_path_with_type(0, 20, 10, 0),
                    _mk_path_with_type(0, 21, 11, 0),
                    _mk_path_with_type(0, 22, 12, 0),
                    _mk_path_with_type(0, 23, 13, 0),
                    _mk_path_with_type(0, 24, 14, 0),
                    _mk_path_with_type(1, 30, 20, 0),
                    _mk_path_with_type(1, 31, 21, 1),
                    _mk_path_with_type(1, 32, 22, 2),
                    _mk_path_with_type(1, 33, 23, 3),
                ],
                "expected": ((1 / 4) + 1.0) / 2,
            },
        ]

        got = []
        expected = []

        metric_dict = None
        for c in cases:
            metric_dict = Metric.calculate_metric(
                _DummyDataObject(paths_value=c["paths"], max_path_type=max_path_type)
            )

            # PTD is k-invariant in current implementation: same scalar copied for each k
            values = []
            for k in config["topk"]:
                values.append(round(float(metric_dict[f"PTD@{k}"]), dp))

            got.append(values[0])
            expected.append(round(float(c["expected"]), dp))

            # Extra guard: all PTD@k must be identical (documents current behavior)
            self.assertEqual(len(set(values)), 1)

        self.assertEqual(got, expected)


class TestExplainabilityLIR(unittest.TestCase):
    def test_lir(self):
        """
        LIR (Linking Interaction Recency):

        Implementation behavior (from hopwise/evaluator/metrics.py):
        - Build per-user LIR values from timestamp_matrix via EMA + min-max normalization
        - For each path, take the LIR of the linking interaction item (path[1][-1]),
          also accepting (path[1][1] == "entity" and id < num_items) as an item
        - Average over all selected paths, then copy the same scalar for every k

        Edge-case focus (formula/semantics):
        - with exactly two interactions per user, normalized LIR becomes {0, 1} deterministically
        - linking item not present in history -> LIR=0
        - entity-typed linking interaction with id < num_items must be accepted
        - verify that aggregation is an average over PATHS (not per-user average)
        """
        name = "lir"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]

        # LIR reads these
        num_items = 6

        def _mk_path_link_item(user, rec_item, linking_item_id, linking_type="item"):
            """
            Build a path where the linking interaction node type can be 'item' or 'entity'.
            LIR will accept:
              - type == "item"
              - type == "entity" AND id < num_items
            """
            path = [
                (0, "user", user),
                (1, linking_type, linking_item_id),  # <- linking interaction (li_idx=1)
                (2, "entity", 999),
                (3, "item", rec_item),
            ]
            return (user, rec_item, 0.0, path)

        # Timestamp matrix shape: (num_users, num_items)
        # We craft EXACTLY 2 interactions per user, so after min-max normalization:
        # older interaction -> 0, more recent -> 1 (deterministic).
        #
        # user0 interacted with item1 at t=100 (older) and item3 at t=200 (recent)
        # user1 interacted with item0 at t=50  (older) and item2 at t=150 (recent)
        timestamp_matrix = np.array(
            [
                [0, 100, 0, 200, 0, 0],  # user0
                [50, 0, 150, 0, 0, 0],   # user1
            ],
            dtype=np.float32,
        )

        cases = [
            {
                "name": "one_user_only_old_linking_interactions",
                # user0 links always to item1 (older) -> LIR = 0, avg -> 0
                "paths": [
                    _mk_path_link_item(0, 20, 1, "item"),
                    _mk_path_link_item(0, 21, 1, "item"),
                    _mk_path_link_item(0, 22, 1, "item"),
                ],
                "expected": 0.0,
            },
            {
                "name": "one_user_only_recent_linking_interactions",
                # user0 links always to item3 (recent) -> LIR = 1, avg -> 1
                "paths": [
                    _mk_path_link_item(0, 20, 3, "item"),
                    _mk_path_link_item(0, 21, 3, "item"),
                ],
                "expected": 1.0,
            },
            {
                "name": "one_user_mixed_old_and_recent",
                # user0 links to {item1 (0), item3 (1)} -> avg over paths: (0 + 1)/2 = 0.5
                "paths": [
                    _mk_path_link_item(0, 20, 1, "item"),
                    _mk_path_link_item(0, 21, 3, "item"),
                ],
                "expected": 0.5,
            },
            {
                "name": "linking_item_not_in_history_must_be_zero",
                # user0 links to item4 which has timestamp 0 (never interacted) -> LIR matrix stays 0
                "paths": [
                    _mk_path_link_item(0, 20, 4, "item"),
                    _mk_path_link_item(0, 21, 4, "item"),
                ],
                "expected": 0.0,
            },
            {
                "name": "entity_typed_linking_id_lt_num_items_must_be_accepted",
                # linking_type='entity' but id=3 < num_items -> should be treated as item3 -> LIR=1
                "paths": [
                    _mk_path_link_item(0, 20, 3, "entity"),
                ],
                "expected": 1.0,
            },
            {
                "name": "two_users_path_level_average_not_user_average",
                # Paths:
                # - user0 links to recent (item3 -> 1) three times
                # - user1 links to old    (item0 -> 0) one time
                # Path-level avg = (1+1+1+0)/4 = 0.75
                # If someone incorrectly averages per-user first: (1.0 + 0.0)/2 = 0.5 (WRONG)
                "paths": [
                    _mk_path_link_item(0, 20, 3, "item"),
                    _mk_path_link_item(0, 21, 3, "item"),
                    _mk_path_link_item(0, 22, 3, "item"),
                    _mk_path_link_item(1, 30, 0, "item"),
                ],
                "expected": 0.75,
            },
        ]

        for c in cases:
            metric_dict = Metric.calculate_metric(
                _DummyDataObject(
                    paths_value=c["paths"],
                    timestamp_matrix=timestamp_matrix,
                    num_items=num_items,
                )
            )

            # LIR is k-invariant in current implementation: same scalar copied for each k
            values = []
            for k in config["topk"]:
                values.append(round(float(metric_dict[f"lir@{k}"]), dp))

            expected_value = round(float(c["expected"]), dp)
            self.assertTrue(all(v == expected_value for v in values))
            self.assertEqual(len(set(values)), 1)


class TestExplainabilitySEP(unittest.TestCase):
    def test_sep(self):
        """
        SEP (Shared Entity Popularity):

        Implementation behavior (from hopwise/evaluator/metrics.py):
        - Build sep_matrix per node_type by sorting degrees ascending and applying normalized_ema
        - For each path, take shared_entity_id/type from path[-2]
          * if shared_entity_type == "item": treat it as "entity"
        - Average SEP over all selected paths
        - Copy the same scalar for every k (k-invariant output pattern)

        Edge-case focus (formula/semantics):
        - popularity must be computed per-type (entity vs brand must not mix)
        - shared_entity_type "item" must map to "entity"
        - mean over paths must be correct (not per-user mean)
        - output must be a dict with SEP@k keys (catches return-value bugs)
        """
        name = "sep"
        Metric = metrics_dict[name](config)
        dp = config["metric_decimal_place"]
        ks = config["topk"]

        # Node degrees by type (in-degree proxy). We choose exactly 2 nodes per type to make
        # normalized_ema deterministic after min-max normalization: lower degree -> 0, higher -> 1.
        #
        # For 'entity': 900 is less popular than 901
        # For 'brand' : 800 is less popular than 801
        node_degree = {
            "entity": {900: 1, 901: 2},
            "brand":  {800: 5, 801: 10},
        }

        def _mk_path_shared(user, rec_item, linking_item_id, shared_type, shared_id):
            """
            Control shared entity type/id at path[-2].
            SEP reads:
              shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
              if shared_entity_type == "item": shared_entity_type = "entity"
            """
            path = [
                (0, "user", user),
                (1, "item", linking_item_id),
                (2, shared_type, shared_id),   # <- shared entity (type/id)
                (3, "item", rec_item),
            ]
            return (user, rec_item, 0.0, path)

        cases = [
            {
                "name": "all_paths_low_pop_entity",
                # entity 900 -> expected SEP weight 0
                "paths": [
                    _mk_path_shared(0, 20, 10, "entity", 900),
                    _mk_path_shared(0, 21, 11, "entity", 900),
                    _mk_path_shared(0, 22, 12, "entity", 900),
                ],
                "expected": 0.0,
            },
            {
                "name": "mix_low_and_high_entity_mean_over_paths",
                # entity 900 -> 0, entity 901 -> 1 => mean of [0,1,1,0] = 0.5
                "paths": [
                    _mk_path_shared(0, 20, 10, "entity", 900),
                    _mk_path_shared(0, 21, 11, "entity", 901),
                    _mk_path_shared(0, 22, 12, "entity", 901),
                    _mk_path_shared(0, 23, 13, "entity", 900),
                ],
                "expected": 0.5,
            },
            {
                "name": "shared_type_item_must_map_to_entity",
                # shared_type == "item" must be treated as "entity"
                # item->entity mapping means id 901 should yield weight 1.0
                "paths": [
                    _mk_path_shared(0, 20, 10, "item", 901),
                    _mk_path_shared(0, 21, 11, "item", 901),
                ],
                "expected": 1.0,
            },
            {
                "name": "different_types_must_not_mix_distributions",
                # brand 800 -> 0, brand 801 -> 1, entity 901 -> 1
                # mean of [brand0, brand1, entity1] = (0 + 1 + 1)/3 = 0.6667
                "paths": [
                    _mk_path_shared(0, 20, 10, "brand", 800),
                    _mk_path_shared(0, 21, 11, "brand", 801),
                    _mk_path_shared(0, 22, 12, "entity", 901),
                ],
                "expected": (0.0 + 1.0 + 1.0) / 3,
            },
        ]

        for c in cases:
            metric_dict = Metric.calculate_metric(
                _DummyDataObject(
                    paths_value=c["paths"],
                    node_degree=node_degree,
                )
            )

            # --- output contract check (catches implementation returning the wrong thing) ---
            self.assertIsInstance(metric_dict, dict)

            # SEP is k-invariant in current implementation: same scalar copied for each k
            values = []
            for k in ks:
                key = f"SEP@{k}"
                self.assertIn(key, metric_dict)
                values.append(round(float(metric_dict[key]), dp))

            expected_value = round(float(c["expected"]), dp)

            # correctness
            self.assertTrue(all(v == expected_value for v in values))

            # invariance across k (documents current behavior)
            self.assertEqual(len(set(values)), 1)


if __name__ == "__main__":
    unittest.main()
