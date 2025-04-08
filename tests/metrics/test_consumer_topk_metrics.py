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
import torch

from hopwise.config import Config
from hopwise.data.interaction import Interaction
from hopwise.evaluator.register import metrics_dict

parameters_dict = {"topk": [10], "metric_decimal_place": 4, "sensitive_attribute": "gender"}

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

users = torch.LongTensor([2, 1, 4, 3])
user_feat = Interaction(
    {
        "user_id": torch.LongTensor([0, 1, 2, 3, 4]),
        "gender": torch.LongTensor([0, 1, 2, 1, 2]),
    }
)


class TestConsumerTopKMetrics(unittest.TestCase):
    def test_deltahit(self):
        name = "deltahit"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 1) / 2 - (0 + 1) / 2),
                    ]
                ]
            ).tolist(),
        )

    def test_deltandcg(self):
        name = "deltandcg"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        print(ranking_result)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 0) / 2 - ((1 / np.log2(2) / (1 / np.log2(2) + 1 / np.log2(3))) + 0) / 2),
                        np.abs(
                            (1 + (1 / np.log2(4) / (1 / np.log2(2) + 1 / np.log2(3)))) / 2
                            - (
                                (
                                    (1 / np.log2(2) + 1 / np.log2(4))
                                    / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
                                )
                                + 0
                            )
                            / 2  # noqa: E501
                        ),
                    ]
                ]
            ).tolist(),
        )

    def test_deltamrr(self):
        name = "deltamrr"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 1 / 3) / 2 - (0 + 1) / 2),
                    ]
                ]
            ).tolist(),
        )

    def test_deltamap(self):
        name = "deltamap"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 + 0) / 2 - (0 + 1) / 2),
                        np.abs((1 + 0) / 2 - (0 + (1 / 2)) / 2),
                        np.abs((1 + (1 / 3) * (1 / 2)) / 2 - (0 + (1 / 3) * ((1 / 1) + (2 / 3))) / 2),
                    ]
                ]
            ).tolist(),
        )

    def test_deltarecall(self):
        name = "deltarecall"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        print(ranking_result)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 / 3 + 0) / 2 - (0 + 1 / 4) / 2),
                        np.abs((2 / 3 + 0) / 2 - (1 / 4 + 0) / 2),
                        np.abs((3 / 3 + 1 / 2) / 2 - (2 / 4 + 0) / 2),
                    ]
                ]
            ).tolist(),
        )

    def test_deltaprecision(self):
        name = "deltaprecision"
        Metric = metrics_dict[name](config)
        group1_mask, group2_mask = Metric.get_group_mask(user_feat, users)
        ranking_result = Metric.ranking_metric_info(pos_idx, pos_len)
        print(ranking_result)
        self.assertEqual(
            Metric.get_dp(ranking_result, group1_mask, group2_mask).tolist(),
            np.array(
                [
                    [
                        np.abs((1 / 1 + 0) / 2 - (0 + 1 / 1) / 2),
                        np.abs((2 / 2 + 0) / 2 - (1 / 2 + 0) / 2),
                        np.abs((3 / 3 + 1 / 3) / 2 - (2 / 3 + 0) / 2),
                    ]
                ]
            ).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
