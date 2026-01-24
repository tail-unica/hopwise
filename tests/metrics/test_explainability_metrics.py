# @Time    : 2026/01/24
# @Author  : Generated for HopWise explainability metric unit test (Fidelity only)

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

K = 5  # we want Fidelity@5 behavior in this unit test
config["topk"] = [K]  # ensure the metric normalizes by K

# Build paths in the format expected by hopwise:
# each entry must be (user, item, score, path)
# An "unexplainable" recommendation is represented by a missing entry for that item.
paths = [
    (0, 20, 0.0, ["U:1", "R:watched", "I:10", "R:directed", "E:director:3", "R:directed", "I:20"]),
    (0, 21, 0.0, ["U:1", "R:watched", "I:11", "R:starred",  "E:actor:7",    "R:starred",  "I:21"]),
    (0, 22, 0.0, ["U:1", "R:clicked", "I:10", "R:belongs_to", "E:genre:2",   "R:belongs_to", "I:22"]),
    (0, 23, 0.0, ["U:1", "R:watched", "I:12", "R:directed", "E:director:3", "R:directed", "I:23"]),
    # 5th recommended item is unexplainable -> not present in `paths`
]


class _DummyDataObject:
    """Minimal dataobject stub: Fidelity only needs `rec.paths` via PathQualityMetric.used_info()."""

    def __init__(self, paths_value):
        self._paths_value = paths_value

    def get(self, key):
        if key == "rec.paths":
            return self._paths_value
        raise KeyError(key)


class TestExplainabilityFID(unittest.TestCase):
    def test_fidelity(self):
        name = "fidelity"
        Metric = metrics_dict[name](config)

        dataobject = _DummyDataObject(paths)
        out = Metric.calculate_metric(dataobject)

        # Fidelity returns a dict with key "Fidelity@K" (capital F) from topk_result in the metric code.
        expected_value = 4 / K  # 4 explainable items out of top-5
        self.assertIn(f"Fidelity@{K}", out)
        self.assertEqual(float(out[f"Fidelity@{K}"]), float(round(expected_value, config["metric_decimal_place"])))


if __name__ == "__main__":
    unittest.main()
