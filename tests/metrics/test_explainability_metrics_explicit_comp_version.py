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

class _NumpyWrap:
    def __init__(self, arr):
        self._arr = np.array(arr)
    def numpy(self):
        return self._arr


class _DummyDataObject:
    """
    Minimal dataobject stub for hopwise metrics.
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
        # --- added for beyond-utility metrics (Serendipity/Novelty) ---
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


class TestExplainabilitySED(unittest.TestCase):
    def test_sed(self):
        name = "sed"
        Metric = metrics_dict[name](config)
        key = f"SED@{K}"
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
                    # shared entity ids are always the same (entity=999) -> unique shared entities = 1
                    # paths = 4 => 1/4
                    1 / 4,

                    # one_user_all_explainable_all_distinct:
                    # unique shared entities = 1, paths = 5 => 1/5
                    1 / 5,

                    # one_user_sparse_all_same_linking:
                    # unique shared entities = 1, paths = 2 => 1/2
                    1 / 2,

                    # two_users_mixed:
                    # user0: unique shared entities = 1, paths = 3 => 1/3
                    # user1: unique shared entities = 1, paths = 3 => 1/3
                    # average = ((1/3) + (1/3)) / 2
                    ((1 / 3) + (1 / 3)) / 2,
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilityLITD(unittest.TestCase):
    def test_litd(self):
        name = "litd"
        Metric = metrics_dict[name](config)
        key = f"LITD@{K}"
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
                    1 / 4,                        # one_user_basic -> 1 / n_paths
                    1 / 5,                        # one_user_all_explainable_all_distinct -> 1 / n_paths
                    1 / 2,                        # one_user_sparse_all_same_linking -> 1 / n_paths
                    ((1 / 3) + (1 / 3)) / 2,      # two_users_mixed -> average of users
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilitySETD(unittest.TestCase):
    def test_setd(self):
        name = "setd"
        Metric = metrics_dict[name](config)
        key = f"SETD@{K}"
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
                    # shared entity ids across paths are the same -> unique shared entities = 1
                    # paths = 4 -> 1/4
                    1 / 4,

                    # one_user_all_explainable_all_distinct:
                    # unique shared entities = 1
                    # paths = 5 -> 1/5
                    1 / 5,

                    # one_user_sparse_all_same_linking:
                    # unique shared entities = 1
                    # paths = 2 -> 1/2
                    1 / 2,

                    # two_users_mixed:
                    # user0: paths = 3 -> 1/3
                    # user1: paths = 3 -> 1/3
                    # average = ((1/3) + (1/3)) / 2
                    ((1 / 3) + (1 / 3)) / 2,
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilityPTD(unittest.TestCase):
    def test_ptd(self):
        name = "ptd"
        Metric = metrics_dict[name](config)
        key = f"PTD@{K}"
        dp = config["metric_decimal_place"]

        # Required by PTD: len(max_path_type) is used in the denominator
        max_path_type = ["t0", "t1", "t2", "t3"]  # len = 4

        self.assertEqual(
            [
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[0]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[1]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[2]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[3]["paths"], max_path_type=max_path_type))[key]), dp),
            ],
            np.array(
                [
                    1 / 4,                        # one_user_basic: 1 distinct type / min(4,4)
                    1 / 4,                        # one_user_all_explainable_all_distinct: 1 / min(5,4)=1/4
                    1 / 2,                        # one_user_sparse_all_same_linking: 1 / min(2,4)
                    ((1 / 3) + (1 / 3)) / 2,      # two_users_mixed: avg of users
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilityPTC(unittest.TestCase):
    def test_ptc(self):
        name = "ptc"
        Metric = metrics_dict[name](config)
        key = f"PTC@{K}"
        dp = config["metric_decimal_place"]

        # Required by PTC: it builds a counter over `max_path_type` and uses it in the Simpson-like formula.
        # In our _mk_path structure the extracted path_type is the last hop id (e.g., 3), so keep ints.
        max_path_type = [0, 1, 2, 3]  # len = 4, includes the last-hop type used by the paths

        self.assertEqual(
            [
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[0]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[1]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[2]["paths"], max_path_type=max_path_type))[key]), dp),
                round(float(Metric.calculate_metric(_DummyDataObject(CASES[3]["paths"], max_path_type=max_path_type))[key]), dp),
            ],
            np.array(
                [
                    # one_user_basic: N=4, all paths are the same type -> numerator = 4*3, denom = 4*3 -> 1 - 1 = 0
                    0.0,

                    # one_user_all_explainable_all_distinct: N=5, all paths same type -> 1 - (5*4)/(5*4) = 0
                    0.0,

                    # one_user_sparse_all_same_linking: N=2, all paths same type -> 1 - (2*1)/(2*1) = 0
                    0.0,

                    # two_users_mixed: each user N=3, all paths same type -> 0 and 0, average -> 0
                    0.0,
                ]
            ).round(dp).tolist(),
        )


class TestExplainabilityPPT(unittest.TestCase):
    def test_ppt(self):
        name = "ppt"
        Metric = metrics_dict[name](config)
        key = f"PPT@{K}"
        dp = config["metric_decimal_place"]

        max_path_length = 4
        rid2relation = ["user", "item", "entity", "item"]

        self.assertEqual(
            [round(float(Metric.calculate_metric(_DummyDataObject(CASES[0]["paths"], max_path_length=max_path_length, rid2relation=rid2relation))[key]), dp),
             round(float(Metric.calculate_metric(_DummyDataObject(CASES[1]["paths"], max_path_length=max_path_length, rid2relation=rid2relation))[key]), dp),
             round(float(Metric.calculate_metric(_DummyDataObject(CASES[2]["paths"], max_path_length=max_path_length, rid2relation=rid2relation))[key]), dp),
             round(float(Metric.calculate_metric(_DummyDataObject(CASES[3]["paths"], max_path_length=max_path_length, rid2relation=rid2relation))[key]), dp)],
            np.array(
                [
                    # one_user_basic:
                    # tutti i path hanno lo stesso pattern relazionale
                    # pattern distinti = 1
                    # n_paths = 4, max_path_length = 4 -> denom = min(4,4)=4
                    1 / 4,

                    # one_user_all_explainable_all_distinct:
                    # pattern distinti = 1
                    # n_paths = 5, max_path_length = 4 -> denom = min(5,4)=4
                    1 / 4,

                    # one_user_sparse_all_same_linking:
                    # pattern distinti = 1
                    # n_paths = 2, max_path_length = 4 -> denom = min(2,4)=2
                    1 / 2,

                    # two_users_mixed:
                    # user0: n_paths=3 -> 1/min(3,4)=1/3
                    # user1: n_paths=3 -> 1/3
                    # average = ((1/3)+(1/3))/2 = 1/3
                    ((1 / 3) + (1 / 3)) / 2,
                ]
            ).round(dp).tolist(),
        )


class TestBeyondUtilitySerendipity(unittest.TestCase):
    def test_serendipity(self):
        name = "serendipity"
        Metric = metrics_dict[name](config)
        key = f"serendipity@{K}"
        dp = config["metric_decimal_place"]

        max_k = K  # ensure rec_items width matches the only k tested

        num_items = 10
        count_items = {i: (num_items - i) for i in range(num_items)}  # item 0 most popular

        rec_items = [
            [(3 + i) % num_items for i in range(max_k)],  # user 0
            [(6 + i) % num_items for i in range(max_k)],  # user 1
        ]

        num_users = len(rec_items) + 1  # includes padding row 0
        history_index = np.array([[1, 2], [0, 1]], dtype=int)

        dataobject = _DummyDataObject(paths_value=None, rec_items=rec_items, count_items=count_items, num_items=num_items, num_users=num_users, history_index=history_index)

        self.assertEqual(
            [round(float(Metric.calculate_metric(dataobject)[key]), dp)],
            np.array(
                [
                    # SER@K (here K=5):
                    # metric output is 0.6, i.e., 3/5
                    3 / 5,
                ]
            ).round(dp).tolist(),
        )


class TestBeyondUtilityNovelty(unittest.TestCase):
    def test_novelty(self):
        name = "novelty"
        Metric = metrics_dict[name](config)
        key = f"novelty@{K}"
        dp = config["metric_decimal_place"]

        # In questo setup testiamo SOLO novelty@K
        max_k = K

        # Numero totale di item nel catalogo
        num_items = 10

        # Popolarità degli item:
        # item 0 -> count = 1 (meno popolare)
        # item 9 -> count = 10 (più popolare)
        count_items = {i: (i + 1) for i in range(num_items)}

        # Raccomandazioni per ciascun utente (lunghezza = K)
        # user 0 riceve item poco popolari
        # user 1 riceve item mediamente popolari
        rec_items = [
            [(0 + i) % num_items for i in range(max_k)],  # user 0: 0,1,2,3,4,...
            [(3 + i) % num_items for i in range(max_k)],  # user 1: 3,4,5,6,7,...
        ]

        # Dummy dataobject (alcuni campi non sono usati da NOV ma servono allo stub)
        dataobject = _DummyDataObject(
            paths_value=None,
            rec_items=rec_items,
            count_items=count_items,
            num_items=num_items,
            num_users=len(rec_items) + 1,
            history_index=np.empty((2, 0), dtype=int),
        )

        # 1) Normalizzazione della popolarità
        #    min_pop = 1
        #    max_pop = 10
        #    denom = max_pop - min_pop = 9
        #
        #    normalized_pop(i) = (count(i) - 1) / 9 = i / 9
        #
        # 2) Novelty di un item:
        #    novelty(i) = 1 - normalized_pop(i) = 1 - (i / 9)
        #
        # 3) Novelty per utente = media delle novelty dei K item raccomandati
        #
        # 4) Novelty finale = media delle novelty dei singoli utenti
        self.assertEqual(
            [round(float(Metric.calculate_metric(dataobject)[key]), dp)],
            np.array(
                [
                    (
                        # user 0:
                        # item: 0,1,2,3,4
                        # novelty:
                        # 1 - 0/9 = 1
                        # 1 - 1/9 = 8/9
                        # 1 - 2/9 = 7/9
                        # 1 - 3/9 = 6/9
                        # 1 - 4/9 = 5/9
                        ((1 + (8 / 9) + (7 / 9) + (6 / 9) + (5 / 9)) / 5
                         +
                         # user 1:
                         # item: 3,4,5,6,7
                         # novelty:
                         # 1 - 3/9 = 6/9
                         # 1 - 4/9 = 5/9
                         # 1 - 5/9 = 4/9
                         # 1 - 6/9 = 3/9
                         # 1 - 7/9 = 2/9
                         ((6 / 9) + (5 / 9) + (4 / 9) + (3 / 9) + (2 / 9)) / 5)
                        / 2  # media sui due utenti
                    ),
                ]
            ).round(dp).tolist(),
        )


if __name__ == "__main__":
    unittest.main()