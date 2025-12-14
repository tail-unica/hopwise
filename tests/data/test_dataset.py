# @Time   : 2021/1/3
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/3, 2021/7/1, 2021/7/11, 2022/7/10
# @Author  :   Yushuo Chen, Xingyu Pan, Yupeng Hou, Lanling Xu
# @email   :   chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn, xulanling_sherry@163.com

import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from hopwise.config import Config
from hopwise.data import create_dataset
from hopwise.utils import PathLanguageModelingTokenType, init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataset(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    return create_dataset(config)


def split_dataset(config_dict=None, config_file_list=None):
    dataset = new_dataset(config_dict=config_dict, config_file_list=config_file_list)
    return dataset.build()


class TestDataset:
    def test_filter_nan_user_or_item(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_nan_user_or_item",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1
        assert len(dataset.user_feat) == 3
        assert len(dataset.item_feat) == 3

    def test_remove_duplication_by_first(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remove_duplication",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 0

    def test_remove_duplication_by_last(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remove_duplication",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "last",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.inter_feat[dataset.time_field][0] == 2

    def test_filter_by_field_value_with_lowest_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_with_highest_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_by_field_value_with_equal_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "[0,0]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 3

    def test_filter_by_field_value_with_not_equal_val(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "(-inf,4);(4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 9

    def test_filter_by_field_value_in_same_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[3,8]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 6

    def test_filter_by_field_value_in_different_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_field_value",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "timestamp": "[3,8]",
                "rating": "(-inf,4);(4,inf)",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 5

    def test_filter_inter_by_user_or_item_is_true(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_user_or_item",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_filter_inter_by_user_or_item_is_false(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_user_or_item",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_filter_by_inter_num_in_min_user_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_min_item_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 7
        assert dataset.item_num == 6

    def test_filter_by_inter_num_in_max_user_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 6
        assert dataset.item_num == 7

    def test_filter_by_inter_num_in_max_item_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "item_inter_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_min_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 5
        assert dataset.item_num == 5

    def test_filter_by_inter_num_in_complex_way(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "user_inter_num_interval": "[2,3]",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_rm_dup_by_first_and_filter_value(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_value",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
            "val_interval": {
                "rating": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 1

    def test_rm_dup_by_last_and_filter_value(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_value",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "last",
            "val_interval": {
                "rating": "(-inf,4]",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2

    def test_rm_dup_and_filter_by_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "rm_dup_and_filter_by_inter_num",
            "data_path": current_path,
            "load_col": None,
            "rm_dup_inter": "first",
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_filter_inter_by_ui(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_value_and_filter_inter_by_ui",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "age": "(-inf,2]",
                "price": "(-inf,2);(2,inf)",
            },
            "filter_inter_by_user_or_item": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 2
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_value_and_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_value_and_inter_num",
            "data_path": current_path,
            "load_col": None,
            "val_interval": {
                "rating": "(-inf,0]",
                "age": "(-inf,0]",
                "price": "(-inf,0]",
            },
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_filter_inter_by_ui_and_inter_num(self):
        config_dict = {
            "model": "BPR",
            "dataset": "filter_inter_by_ui_and_inter_num",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": True,
            "user_inter_num_interval": "[2,inf)",
            "item_inter_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert len(dataset.inter_feat) == 4
        assert dataset.user_num == 3
        assert dataset.item_num == 3

    def test_remap_id(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remap_id",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id("user_id", ["ua", "ub", "uc", "ud"])
        item_list = dataset.token2id("item_id", ["ia", "ib", "ic", "id"])
        assert (user_list == [1, 2, 3, 4]).all()
        assert (item_list == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_user"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_item"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["user_list"][0] == [1, 2]).all()
        assert (dataset.inter_feat["user_list"][1] == []).all()
        assert (dataset.inter_feat["user_list"][2] == [3, 4, 1]).all()
        assert (dataset.inter_feat["user_list"][3] == [5]).all()

    def test_remap_id_with_alias(self):
        config_dict = {
            "model": "BPR",
            "dataset": "remap_id",
            "data_path": current_path,
            "load_col": None,
            "alias_of_user_id": ["add_user", "user_list"],
            "alias_of_item_id": ["add_item"],
        }
        dataset = new_dataset(config_dict=config_dict)
        user_list = dataset.token2id("user_id", ["ua", "ub", "uc", "ud", "ue", "uf"])
        item_list = dataset.token2id("item_id", ["ia", "ib", "ic", "id", "ie", "if"])
        assert (user_list == [1, 2, 3, 4, 5, 6]).all()
        assert (item_list == [1, 2, 3, 4, 5, 6]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3, 4]).all()
        assert (dataset.inter_feat["add_user"] == [2, 5, 4, 6]).all()
        assert (dataset.inter_feat["add_item"] == [5, 3, 6, 1]).all()
        assert (dataset.inter_feat["user_list"][0] == [3, 5]).all()
        assert (dataset.inter_feat["user_list"][1] == []).all()
        assert (dataset.inter_feat["user_list"][2] == [1, 2, 3]).all()
        assert (dataset.inter_feat["user_list"][3] == [6]).all()

    def test_ui_feat_preparation_and_fill_nan(self):
        config_dict = {
            "model": "BPR",
            "dataset": "ui_feat_preparation_and_fill_nan",
            "data_path": current_path,
            "load_col": None,
            "filter_inter_by_user_or_item": False,
            "normalize_field": None,
            "normalize_all": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        user_token_list = dataset.id2token("user_id", dataset.user_feat["user_id"])
        item_token_list = dataset.id2token("item_id", dataset.item_feat["item_id"])
        assert (user_token_list == ["[PAD]", "ua", "ub", "uc", "ud", "ue"]).all()
        assert (item_token_list == ["[PAD]", "ia", "ib", "ic", "id", "ie"]).all()
        assert dataset.inter_feat["rating"][3] == 1.0
        assert dataset.user_feat["age"][4] == 1.5
        assert dataset.item_feat["price"][4] == 1.5
        assert (dataset.inter_feat["time_list"][0] == [1.0, 2.0, 3.0]).all()
        assert (dataset.inter_feat["time_list"][1] == [2.0]).all()
        assert (dataset.inter_feat["time_list"][2] == []).all()
        assert (dataset.inter_feat["time_list"][3] == [5, 4]).all()
        assert (dataset.user_feat["profile"][0] == []).all()
        assert (dataset.user_feat["profile"][1] == [1, 2, 3]).all()
        assert (dataset.user_feat["profile"][2] == []).all()
        assert (dataset.user_feat["profile"][3] == [3]).all()
        assert (dataset.user_feat["profile"][4] == []).all()
        assert (dataset.user_feat["profile"][5] == [3, 2]).all()

    def test_set_label_by_threshold(self):
        config_dict = {
            "model": "BPR",
            "dataset": "set_label_by_threshold",
            "data_path": current_path,
            "load_col": None,
            "threshold": {
                "rating": 4,
            },
            "normalize_field": None,
            "normalize_all": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["label"] == [1.0, 0.0, 1.0, 0.0]).all()

    def test_normalize_all(self):
        config_dict = {
            "model": "BPR",
            "dataset": "normalize",
            "data_path": current_path,
            "load_col": None,
            "normalize_all": True,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["rating"] == [0.0, 0.25, 1.0, 0.75, 0.5]).all()
        assert (dataset.inter_feat["star"] == [1.0, 0.5, 0.0, 0.25, 0.75]).all()

    def test_normalize_field(self):
        config_dict = {
            "model": "BPR",
            "dataset": "normalize",
            "data_path": current_path,
            "load_col": None,
            "normalize_field": ["rating"],
            "normalize_all": False,
        }
        dataset = new_dataset(config_dict=config_dict)
        assert (dataset.inter_feat["rating"] == [0.0, 0.25, 1.0, 0.75, 0.5]).all()
        assert (dataset.inter_feat["star"] == [4.0, 2.0, 0.0, 1.0, 3.0]).all()

    def test_TO_RS_811(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1]
            + [1, 2, 3]
            + list(range(1, 8))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy() == list(range(17, 19)) + [] + [] + [2] + [4] + [8] + [9] + [10]
        ).all()
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(19, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_TO_RS_820(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.2, 0.0]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(17, 21)) + [] + [2] + [3] + [5] + [9] + [9, 10] + [10, 11]
        ).all()
        assert len(test_dataset.inter_feat) == 0

    def test_TO_RS_802(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.0, 0.2]},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 17))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert len(valid_dataset.inter_feat) == 0
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(17, 21)) + [] + [2] + [3] + [5] + [9] + [9, 10] + [10, 11]
        ).all()

    def test_TO_LS_valid_and_test(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 19))
            + [1]
            + [1]
            + [1]
            + [1, 2, 3]
            + list(range(1, 8))
            + list(range(1, 9))
            + list(range(1, 10))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy() == list(range(19, 20)) + [] + [] + [2] + [4] + [8] + [9] + [10]
        ).all()
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_TO_LS_valid_only(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "valid_only"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 20))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 10))
            + list(range(1, 11))
        ).all()
        assert (
            valid_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()
        assert len(test_dataset.inter_feat) == 0

    def test_TO_LS_test_only(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"LS": "test_only"},
                "order": "TO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (
            train_dataset.inter_feat["item_id"].numpy()
            == list(range(1, 20))
            + [1]
            + [1]
            + [1, 2]
            + [1, 2, 3, 4]
            + list(range(1, 9))
            + list(range(1, 10))
            + list(range(1, 11))
        ).all()
        assert len(valid_dataset.inter_feat) == 0
        assert (
            test_dataset.inter_feat["item_id"].numpy()
            == list(range(20, 21)) + [] + [2] + [3] + [5] + [9] + [10] + [11]
        ).all()

    def test_RO_RS_811(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 1 + 3 + 7 + 8 + 9
        assert len(valid_dataset.inter_feat) == 2 + 0 + 0 + 1 + 1 + 1 + 1 + 1
        assert len(test_dataset.inter_feat) == 2 + 0 + 1 + 1 + 1 + 1 + 1 + 1

    def test_RO_RS_820(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.2, 0.0]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 2 + 4 + 8 + 8 + 9
        assert len(valid_dataset.inter_feat) == 4 + 0 + 1 + 1 + 1 + 1 + 2 + 2
        assert len(test_dataset.inter_feat) == 0

    def test_RO_RS_802(self):
        config_dict = {
            "model": "BPR",
            "dataset": "build_dataset",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.0, 0.2]},
                "order": "RO",
                "mode": "labeled",
            },
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert len(train_dataset.inter_feat) == 16 + 1 + 1 + 2 + 4 + 8 + 8 + 9
        assert len(valid_dataset.inter_feat) == 0
        assert len(test_dataset.inter_feat) == 4 + 0 + 1 + 1 + 1 + 1 + 2 + 2


class TestSeqDataset:
    def test_seq_leave_one_out(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat[train_dataset.uid_field].numpy() == [1, 1, 1, 1, 1, 4, 2, 2, 3]).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :5].numpy()
            == [
                [1, 0, 0, 0, 0],
                [1, 2, 0, 0, 0],
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 0],
                [1, 2, 3, 4, 5],
                [3, 0, 0, 0, 0],
                [4, 0, 0, 0, 0],
                [4, 5, 0, 0, 0],
                [4, 0, 0, 0, 0],
            ]
        ).all()
        assert (train_dataset.inter_feat[train_dataset.iid_field].numpy() == [2, 3, 4, 5, 6, 4, 5, 6, 5]).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_list_length_field].numpy() == [1, 2, 3, 4, 5, 1, 1, 2, 1]
        ).all()

        assert (valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 2]).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :6].numpy()
            == [[1, 2, 3, 4, 5, 6], [4, 5, 6, 0, 0, 0]]
        ).all()
        assert (valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [7, 7]).all()
        assert (valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy() == [6, 3]).all()

        assert (test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 2, 3]).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [[1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]]
        ).all()
        assert (test_dataset.inter_feat[test_dataset.iid_field].numpy() == [8, 8, 6]).all()
        assert (test_dataset.inter_feat[test_dataset.item_list_length_field].numpy() == [7, 4, 2]).all()

        assert (
            train_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()
        assert (
            valid_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()
        assert (
            test_dataset.inter_matrix().toarray()
            == [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).all()

    def test_seq_split_by_ratio(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_dataset",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "eval_args": {"split": {"RS": [0.3, 0.3, 0.4]}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat[train_dataset.uid_field].numpy() == [1, 1, 1, 4, 2, 2, 3]).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :3].numpy()
            == [
                [1, 0, 0],
                [1, 2, 0],
                [1, 2, 3],
                [3, 0, 0],
                [4, 0, 0],
                [4, 5, 0],
                [4, 0, 0],
            ]
        ).all()
        assert (train_dataset.inter_feat[train_dataset.iid_field].numpy() == [2, 3, 4, 4, 5, 6, 5]).all()
        assert (train_dataset.inter_feat[train_dataset.item_list_length_field].numpy() == [1, 2, 3, 1, 1, 2, 1]).all()

        assert (valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 1, 2]).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :5].numpy()
            == [[1, 2, 3, 4, 0], [1, 2, 3, 4, 5], [4, 5, 6, 0, 0]]
        ).all()
        assert (valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [5, 6, 7]).all()
        assert (valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy() == [4, 5, 3]).all()

        assert (test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 1, 2, 3]).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [
                [1, 2, 3, 4, 5, 6, 0],
                [1, 2, 3, 4, 5, 6, 7],
                [4, 5, 6, 7, 0, 0, 0],
                [4, 5, 0, 0, 0, 0, 0],
            ]
        ).all()
        assert (test_dataset.inter_feat[test_dataset.iid_field].numpy() == [7, 8, 8, 6]).all()
        assert (test_dataset.inter_feat[test_dataset.item_list_length_field].numpy() == [6, 7, 4, 2]).all()

    def test_seq_benchmark(self):
        config_dict = {
            "model": "GRU4Rec",
            "dataset": "seq_benchmark",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": None,
            "benchmark_filename": ["train", "valid", "test"],
            "alias_of_item_id": ["item_id_list"],
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        assert (train_dataset.inter_feat[train_dataset.uid_field].numpy() == [1, 1, 1, 2, 3, 3, 4]).all()
        assert (
            train_dataset.inter_feat[train_dataset.item_id_list_field][:, :3].numpy()
            == [
                [8, 0, 0],
                [8, 1, 0],
                [8, 1, 2],
                [2, 0, 0],
                [3, 0, 0],
                [3, 4, 0],
                [3, 0, 0],
            ]
        ).all()
        assert (train_dataset.inter_feat[train_dataset.iid_field].numpy() == [1, 2, 3, 3, 4, 5, 4]).all()
        assert (train_dataset.inter_feat[train_dataset.item_list_length_field].numpy() == [1, 2, 3, 1, 1, 2, 1]).all()

        assert (valid_dataset.inter_feat[valid_dataset.uid_field].numpy() == [1, 1, 3]).all()
        assert (
            valid_dataset.inter_feat[valid_dataset.item_id_list_field][:, :5].numpy()
            == [[8, 1, 2, 3, 0], [8, 1, 2, 3, 4], [3, 4, 5, 0, 0]]
        ).all()
        assert (valid_dataset.inter_feat[valid_dataset.iid_field].numpy() == [4, 5, 6]).all()
        assert (valid_dataset.inter_feat[valid_dataset.item_list_length_field].numpy() == [4, 5, 3]).all()

        assert (test_dataset.inter_feat[test_dataset.uid_field].numpy() == [1, 1, 3, 4]).all()
        assert (
            test_dataset.inter_feat[test_dataset.item_id_list_field][:, :7].numpy()
            == [
                [8, 1, 2, 3, 4, 5, 0],
                [8, 1, 2, 3, 4, 5, 6],
                [3, 4, 5, 6, 0, 0, 0],
                [3, 4, 0, 0, 0, 0, 0],
            ]
        ).all()
        assert (test_dataset.inter_feat[test_dataset.iid_field].numpy() == [6, 7, 7, 5]).all()
        assert (test_dataset.inter_feat[test_dataset.item_list_length_field].numpy() == [6, 7, 4, 2]).all()


class TestKGDataset:
    def test_kg_remap_id(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_remap_id",
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        item_list = dataset.token2id("item_id", ["ib", "ic", "id"])
        entity_list = dataset.token2id("entity_id", ["eb", "ec", "ed", "ee", "ea"])
        assert (item_list == [1, 2, 3]).all()
        assert (entity_list == [1, 2, 3, 4, 5]).all()
        assert (dataset.inter_feat["user_id"] == [1, 2, 3]).all()
        assert (dataset.inter_feat["item_id"] == [1, 2, 3]).all()
        assert (dataset.kg_feat["head_id"] == [1, 2, 3, 4]).all()
        assert (dataset.kg_feat["tail_id"] == [5, 1, 2, 3]).all()

    def test_kg_reverse_r(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_reverse_r",
            "kg_reverse_r": True,
            "data_path": current_path,
            "load_col": None,
        }
        dataset = new_dataset(config_dict=config_dict)
        relation_list = dataset.token2id("relation_id", ["ra", "rb", "ra_r", "rb_r"])
        assert (relation_list == [1, 2, 5, 6]).all()
        assert dataset.relation_num == 10

    def test_kg_filter_by_triple_num_in_min_entity_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 6
        assert dataset.relation_num == 6

    def test_kg_filter_by_triple_num_in_min_relation_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_max_entity_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "(-inf,3]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 3
        assert dataset.relation_num == 3

    def test_kg_filter_by_triple_num_in_max_relation_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "relation_kg_num_interval": "(-inf,2]",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 6
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_min_kg_num(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[1,inf)",
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5

    def test_kg_filter_by_triple_num_in_complex_way(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "kg_filter_by_triple_num",
            "data_path": current_path,
            "load_col": None,
            "entity_kg_num_interval": "[1,4]",
            "relation_kg_num_interval": "[2,inf)",
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.entity_num == 7
        assert dataset.relation_num == 5

    def test_preload_weight(self):
        config_dict = {
            "model": "KGAT",
            "dataset": "test",
            "data_path": os.path.join(current_path, os.pardir, "test_data"),
            "load_col": {
                "useremb": ["user_embedding_id", "user_embedding"],
                "entityemb": ["entity_embedding_id", "entity_embedding"],
                "relationemb": ["relation_embedding_id", "relation_embedding"],
            },
            "alias_of_user_id": ["user_embedding_id"],
            "alias_of_entity_id": ["entity_embedding_id"],
            "alias_of_relation_id": ["relation_embedding_id"],
            "preload_weight": {
                "user_embedding_id": "user_embedding",
                "entity_embedding_id": "entity_embedding",
                "relation_embedding_id": "relation_embedding",
            },
            "additional_feat_suffix": ["useremb", "entityemb", "relationemb"],
        }
        dataset = new_dataset(config_dict=config_dict)
        assert dataset.useremb_feat is not None
        assert dataset.entityemb_feat is not None
        assert dataset.relationemb_feat is not None

    def test_preload_weight_with_preload_weight_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            shutil.copytree(
                os.path.join(current_path, os.pardir, "test_data"),
                os.path.join(tempdir, "test_data"),
            )
            for preload_weight_feat in ["useremb", "entityemb", "relationemb"]:
                shutil.move(
                    os.path.join(tempdir, "test_data", "test", f"test.{preload_weight_feat}"),
                    os.path.join(tempdir, f"test.{preload_weight_feat}"),
                )

            config_dict = {
                "model": "KGAT",
                "dataset": "test",
                "data_path": os.path.join(tempdir, "test_data"),
                "load_col": {
                    "useremb": ["user_embedding_id", "user_embedding"],
                    "entityemb": ["entity_embedding_id", "entity_embedding"],
                    "relationemb": ["relation_embedding_id", "relation_embedding"],
                },
                "alias_of_user_id": ["user_embedding_id"],
                "alias_of_entity_id": ["entity_embedding_id"],
                "alias_of_relation_id": ["relation_embedding_id"],
                "preload_weight": {
                    "user_embedding_id": "user_embedding",
                    "entity_embedding_id": "entity_embedding",
                    "relation_embedding_id": "relation_embedding",
                },
                "additional_feat_suffix": ["useremb", "entityemb", "relationemb"],
                "preload_weight_path": tempdir,
            }
            dataset = new_dataset(config_dict=config_dict)
            assert dataset.useremb_feat is not None
            assert dataset.entityemb_feat is not None
            assert dataset.relationemb_feat is not None


class TestKGPathDataset(unittest.TestCase):
    def test_kg_valid_path(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "max_paths_per_user": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": False,
                "temporal_causality": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        dataset = new_dataset(config_dict=config_dict)
        user_num = dataset.user_num
        item_num = dataset.item_num
        entity_num = dataset.entity_num
        uids = list(range(user_num))
        iids = list(range(user_num + 1, user_num + item_num + 1))
        eids = list(range(user_num + item_num + 1, user_num + item_num + entity_num + 1))
        self.assertTrue(dataset._check_kg_path((uids[1], iids[1], eids[1], eids[2], eids[3]), user_num, item_num))
        self.assertFalse(
            dataset._check_kg_path((iids[1], eids[1], eids[2], eids[3]), user_num, item_num, check_last_node=True)
        )
        self.assertTrue(dataset._check_kg_path((uids[1], iids[1], eids[1], iids[2], eids[3]), user_num, item_num))
        self.assertFalse(dataset._check_kg_path((uids[1], eids[1], eids[2], eids[3]), user_num, item_num))
        self.assertFalse(
            dataset._check_kg_path(
                (uids[1], iids[1], eids[1], uids[1], eids[3]), user_num, item_num, collaborative_path=False
            )
        )
        self.assertTrue(
            dataset._check_kg_path(
                (uids[1], iids[1], eids[1], uids[1], eids[3]), user_num, item_num, collaborative_path=True
            )
        )

    def test_kg_format_path(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "max_paths_per_user": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": False,
                "temporal_causality": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        dataset1, dataset2 = new_dataset(config_dict), new_dataset(config_dict)
        dataset1.path_token_separator = " -> "
        dataset2.path_token_separator = " ## "
        ui_relation = 5
        user3, user2, pos_iid, entity1, entity2, rec_iid = 3, 2, 6, 12, 13, 7
        full_path = np.array([user3, ui_relation, pos_iid, 1, entity1, 2, entity2, 3, rec_iid])
        path_no_entity = np.array([user3, ui_relation, pos_iid, 4, rec_iid])
        padded_path = np.concatenate([full_path, [-1, -1]])
        collaborative_path = np.array([user3, ui_relation, pos_iid, ui_relation, user2, ui_relation, rec_iid])
        assert dataset1._format_path(full_path) == "U3 -> R5 -> I2 -> R1 -> E8 -> R2 -> E9 -> R3 -> I3"
        assert dataset1._format_path(path_no_entity) == "U3 -> R5 -> I2 -> R4 -> I3"
        assert dataset1._format_path(padded_path) == "U3 -> R5 -> I2 -> R1 -> E8 -> R2 -> E9 -> R3 -> I3"
        assert dataset1._format_path(collaborative_path) == "U3 -> R5 -> I2 -> R5 -> U2 -> R5 -> I3"
        assert dataset2._format_path(full_path) == "U3 ## R5 ## I2 ## R1 ## E8 ## R2 ## E9 ## R3 ## I3"

    def test_kg_generate_path_weighted_random_walk_unrestricted_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        used_ids = train_dataset.get_user_used_ids()
        paths = []
        for _ in range(5):  # try multiple times to get a valid path due to randomness
            paths.append(train_dataset.generate_user_paths())  # path ids not remapped
        paths = np.vstack(paths)
        paths_used_ids = used_ids[paths[:, 0]]
        assert any((paths[i, -1] not in paths_used_ids[i] and paths[i, -1] != paths[i, 0]) for i in range(len(paths)))

    def test_kg_generate_path_weighted_random_walk_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        potential_paths = np.array(
            [
                [2, 5, 6, 1, 11, 1, 7],  # ub->[UI-Relation]->eb->ra->ei->ra->ec
                [2, 5, 7, 1, 11, 1, 6],  # reverse of above
                [3, 5, 7, 1, 12, 1, 8],  # uc->[UI-Relation]->ec->ra->ej->ra->ed
                [3, 5, 8, 1, 12, 1, 7],  # reverse of above
                [2, 5, 6, 2, 13, 3, 7],  # ub->[UI-Relation]->eb->rb->ek->rc->ec
                [2, 5, 7, 3, 13, 2, 6],  # reverse of above
                [3, 5, 7, 2, 14, 3, 8],  # uc->[UI-Relation]->ec->rb->el->rc->ed
                [3, 5, 8, 3, 14, 2, 7],  # reverse of above
            ]
        )
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = []
        for _ in range(5):  # try multiple times to get a valid path due to randomness
            paths.append(train_dataset.generate_user_paths())  # path ids not remapped
        paths = np.vstack(paths)
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_weighted_random_walk_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": True,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user_num = train_dataset.user_num
        paths = []
        for _ in range(5):  # try multiple times to get a valid path due to randomness
            paths.append(train_dataset.generate_user_paths())  # path ids not remapped
        paths = np.vstack(paths)
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_weighted_random_walk_no_collaborative_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "weighted-rw",
                "collaborative_path": False,
                "temporal_causality": True,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user = train_dataset.inter_feat[train_dataset.uid_field].numpy()
        item = train_dataset.inter_feat[train_dataset.iid_field].numpy()
        timestamp = train_dataset.inter_feat[train_dataset.time_field].numpy()
        temporal_matrix = np.zeros((train_dataset.user_num, train_dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        paths = []
        for _ in range(5):  # try multiple times to get a valid path due to randomness
            paths.append(train_dataset.generate_user_paths())  # path ids not remapped
        paths = np.vstack(paths)
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - train_dataset.user_num
        subsequent_pos_items = paths[:, -1] - train_dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_generate_path_constrained_random_walk_unrestricted_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "constrained-rw",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        used_ids = train_dataset.get_user_used_ids()
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        paths_used_ids = used_ids[paths[:, 0]]
        assert any((paths[i, -1] not in paths_used_ids[i] and paths[i, -1] != paths[i, 0]) for i in range(len(paths)))

    def test_kg_generate_path_constrained_random_walk_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "constrained-rw",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        potential_paths = np.array(
            [
                [2, 5, 6, 1, 11, 1, 7],  # ub->[UI-Relation]->eb->ra->ei->ra->ec
                [2, 5, 7, 1, 11, 1, 6],  # reverse of above
                [3, 5, 7, 1, 12, 1, 8],  # uc->[UI-Relation]->ec->ra->ej->ra->ed
                [3, 5, 8, 1, 12, 1, 7],  # reverse of above
                [2, 5, 6, 2, 13, 3, 7],  # ub->[UI-Relation]->eb->rb->ek->rc->ec
                [2, 5, 7, 3, 13, 2, 6],  # reverse of above
                [3, 5, 7, 2, 14, 3, 8],  # uc->[UI-Relation]->ec->rb->el->rc->ed
                [3, 5, 8, 3, 14, 2, 7],  # reverse of above
            ]
        )
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_constrained_random_walk_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "constrained-rw",
                "collaborative_path": True,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user_num = train_dataset.user_num
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_constrained_random_walk_no_collaborative_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "constrained-rw",
                "collaborative_path": False,
                "temporal_causality": True,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user = train_dataset.inter_feat[train_dataset.uid_field].numpy()
        item = train_dataset.inter_feat[train_dataset.iid_field].numpy()
        timestamp = train_dataset.inter_feat[train_dataset.time_field].numpy()
        temporal_matrix = np.zeros((train_dataset.user_num, train_dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - train_dataset.user_num
        subsequent_pos_items = paths[:, -1] - train_dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_generate_path_simple_ui_unrestricted_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 100,
            "path_sample_args": {
                "strategy": "simple-ui",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        possible_valid_paths = {
            # User 1 paths
            (1, 5, 6, 1, 11, 1, 7),  # ua: eb(6)→ra→ei(11)→ra→ec(7)
            (1, 5, 6, 2, 13, 3, 7),  # ua: eb(6)→rb→ek(13)→rc→ec(7)
            # User 2 paths
            (2, 5, 6, 1, 11, 1, 7),  # ub: eb(6)→ra→ei(11)→ra→ec(7)
            (2, 5, 7, 1, 11, 1, 6),  # ub: ec(7)→ra→ei(11)→ra→eb(6)
            (2, 5, 6, 2, 13, 3, 7),  # ub: eb(6)→rb→ek(13)→rc→ec(7)
            (2, 5, 7, 1, 12, 1, 8),  # ub: ec(7)→ra→ej(12)→ra→ed(8)
            (2, 5, 7, 2, 14, 3, 8),  # ub: ec(7)→rb→el(14)→rc→ed(8)
            # User 3 paths
            (3, 5, 7, 1, 11, 1, 6),  # uc: ec(7)→ra→ei(11)→ra→eb(6)
            (3, 5, 7, 1, 12, 1, 8),  # uc: ec(7)→ra→ej(12)→ra→ed(8)
            (3, 5, 7, 2, 14, 3, 8),  # uc: ec(7)→rb→el(14)→rc→ed(8)
            # ed(8) has no outgoing edges, so no paths starting from id(8)
        }

        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.generate_user_paths()
        valid_paths = paths[paths[:, -1] != -1]
        found_paths = set(tuple(p) for p in valid_paths)

        # All found paths must be in the set of possible valid paths
        invalid_paths = found_paths - possible_valid_paths
        assert len(invalid_paths) == 0, f"Found invalid paths: {invalid_paths}"

        # With unrestricted mode, we expect paths that reach items outside positive items
        # At minimum, some paths should be found
        assert len(found_paths) > 0, "No paths found, but valid paths should exist"

    def test_kg_generate_path_simple_ui_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 100,
            "path_sample_args": {
                "strategy": "simple-ui",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        possible_valid_paths = {
            (2, 5, 6, 1, 11, 1, 7),  # ub: ib(6)->ra->ei(11)->ra->ic(7)
            (2, 5, 7, 1, 11, 1, 6),  # ub: ic(7)->ra->ei(11)->ra->ib(6)  (reverse exists in KG)
            (2, 5, 6, 2, 13, 3, 7),  # ub: ib(6)->rb->ek(13)->rc->ic(7)
            (3, 5, 7, 1, 12, 1, 8),  # uc: ic(7)->ra->ej(12)->ra->id(8)
            (3, 5, 7, 2, 14, 3, 8),  # uc: ic(7)->rb->el(14)->rc->id(8)
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        used_ids = train_dataset.get_user_used_ids()
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        # Filter out padding values
        valid_paths = paths[paths[:, -1] != -1]

        # Verify that all found paths are valid (subset of possible paths)
        found_paths = set(tuple(p) for p in valid_paths)
        assert found_paths.issubset(possible_valid_paths), f"Found invalid paths: {found_paths - possible_valid_paths}"

        # Verify some paths were found (since simple-ui uses sampling, not exhaustive search)
        assert len(found_paths) > 0, "No paths found, but valid paths exist"

        # User 1 should have NO paths (ia and ib are not connected via 2-hop entity paths)
        user1_paths = valid_paths[valid_paths[:, 0] == 1]
        assert len(user1_paths) == 0, f"User 1 should have no paths, but found {len(user1_paths)}"

        # User 2 should have paths - verify they end in user's positive items
        user2_paths = valid_paths[valid_paths[:, 0] == 2]
        user2_pos_items_graph = {iid + train_dataset.user_num for iid in used_ids[2] if iid != 0}
        reached_items = set(user2_paths[:, -1])
        assert reached_items.issubset(user2_pos_items_graph)

        # User 3 should have paths (but only starting from ic, not from id due to directed KG)
        user3_paths = valid_paths[valid_paths[:, 0] == 3]
        user3_pos_items_graph = {iid + train_dataset.user_num for iid in used_ids[3] if iid != 0}
        reached_items = set(user3_paths[:, -1])
        assert reached_items.issubset(user3_pos_items_graph)

    def test_kg_generate_path_simple_ui_collaborative_no_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 100,
            "path_sample_args": {
                "strategy": "simple-ui",
                "collaborative_path": True,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        possible_valid_paths = {
            # User 1 (ua=1): items {ia/ea=5, ib/eb=6}
            # KG paths: ia has no 2-hop path to ib via entities
            # Collaborative: ia→ua→ib exists via UI-Relation
            (1, 5, 5, 5, 1, 5, 6),  # ua: ia(5)→ui_rel→ua(1)→ui_rel→ib(6)
            (1, 5, 6, 5, 1, 5, 5),  # ua: ib(6)→ui_rel→ua(1)→ui_rel→ia(5)
            # User 2 (ub=2): items {ib/eb=6, ic/ec=7}
            # KG paths:
            (2, 5, 6, 1, 11, 1, 7),  # ub: eb(6)→ra→ei(11)→ra→ec(7)
            (2, 5, 7, 1, 11, 1, 6),  # ub: ec(7)→ra→ei(11)→ra→eb(6)
            (2, 5, 6, 2, 13, 3, 7),  # ub: eb(6)→rb→ek(13)→rc→ec(7)
            # Collaborative:
            (2, 5, 6, 5, 2, 5, 7),  # ub: eb(6)→ui_rel→ub(2)→ui_rel→ec(7)
            (2, 5, 7, 5, 2, 5, 6),  # ub: ec(7)→ui_rel→ub(2)→ui_rel→eb(6)
            # User 3 (uc=3): items {ic/ec=7, id/ed=8}
            # KG paths:
            (3, 5, 7, 1, 12, 1, 8),  # uc: ec(7)→ra→ej(12)→ra→ed(8)
            (3, 5, 7, 2, 14, 3, 8),  # uc: ec(7)→rb→el(14)→rc→ed(8)
            # ed(8) has no outgoing edges in KG
            # Collaborative:
            (3, 5, 7, 5, 3, 5, 8),  # uc: ec(7)→ui_rel→uc(3)→ui_rel→ed(8)
            (3, 5, 8, 5, 3, 5, 7),  # uc: ed(8)→ui_rel→uc(3)→ui_rel→ec(7)
        }

        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.generate_user_paths()
        valid_paths = paths[paths[:, -1] != -1]
        found_paths = set(tuple(p) for p in valid_paths)

        # All found paths must be valid
        invalid_paths = found_paths - possible_valid_paths
        assert len(invalid_paths) == 0, f"Found invalid paths: {invalid_paths}"

        # Must find at least some paths
        assert len(found_paths) > 0, "No paths found, but valid paths should exist"

        # Check paths end in user's positive items
        used_ids = train_dataset.get_user_used_ids()
        for path in valid_paths:
            user_id = path[0]
            end_item = path[-1]
            user_positive_items_graph = {iid + train_dataset.user_num for iid in used_ids[user_id]}
            assert (
                end_item in user_positive_items_graph
            ), f"Path {tuple(path)} ends at {end_item} which is not in user {user_id}'s positive items"

    def test_kg_generate_path_simple_ui_no_collaborative_temporal(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 100,
            "path_sample_args": {
                "strategy": "simple-ui",
                "collaborative_path": False,
                "temporal_causality": True,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        # Expected valid paths with temporal constraint (start_time < end_time):
        # User 1: ia(t=1)→...→ib(t=2). But ia and ib not connected via 2-hop KG paths
        #         ia has edges to eg, eh (which have no outgoing edges)
        #         So NO valid paths for user 1 with temporal constraint.
        # User 2: ib(t=1)→...→ic(t=2). eb→ei→ec and eb→ek→ec exist!
        # User 3: ic(t=1)→...→id(t=2). ec→ej→ed and ec→el→ed exist!
        possible_valid_paths = {
            # User 2: ib(t=1)→ic(t=2)
            (2, 5, 6, 1, 11, 1, 7),  # ub: eb(6)→ra→ei(11)→ra→ec(7)
            (2, 5, 6, 2, 13, 3, 7),  # ub: eb(6)→rb→ek(13)→rc→ec(7)
            # User 3: ic(t=1)→id(t=2)
            (3, 5, 7, 1, 12, 1, 8),  # uc: ec(7)→ra→ej(12)→ra→ed(8)
            (3, 5, 7, 2, 14, 3, 8),  # uc: ec(7)→rb→el(14)→rc→ed(8)
        }

        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.generate_user_paths()
        valid_paths = paths[paths[:, -1] != -1]
        found_paths = set(tuple(p) for p in valid_paths)

        # All found paths must be valid
        invalid_paths = found_paths - possible_valid_paths
        assert len(invalid_paths) == 0, f"Found invalid paths: {invalid_paths}"

        # Some paths should be found (users 2 and 3 have valid temporal paths)
        assert len(found_paths) > 0, "No paths found, but valid paths should exist"

        # User 1 should have NO paths (ia and ib not connected via 2-hop entity paths)
        user1_paths = valid_paths[valid_paths[:, 0] == 1]
        assert len(user1_paths) == 0, "User 1 should have no paths with temporal constraint"

        # Verify temporal constraint: check that start item's timestamp < end item's timestamp
        user = train_dataset.inter_feat[train_dataset.uid_field].numpy()
        item = train_dataset.inter_feat[train_dataset.iid_field].numpy()
        timestamp = train_dataset.inter_feat[train_dataset.time_field].numpy()
        temporal_matrix = np.zeros((train_dataset.user_num, train_dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp

        for path in valid_paths:
            user_id = path[0]
            start_item = path[2] - train_dataset.user_num  # Convert graph ID to item ID
            end_item = path[-1] - train_dataset.user_num
            start_time = temporal_matrix[user_id, start_item]
            end_time = temporal_matrix[user_id, end_item]
            assert start_time < end_time, (
                f"Temporal constraint violated: path {tuple(path)} " f"starts at t={start_time}, ends at t={end_time}"
            )

    def test_kg_generate_path_metapaths_unrestricted_no_collaborative_no_temporal(self):
        metapaths = [[("item_id", "ra", "entity_id"), ("entity_id", "ra", "item_id")], ["rb", "rc"]]
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "metapath",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": False,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        # Differently from igraph, reverse paths must be explicitly defined in the KG
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        used_ids = train_dataset.get_user_used_ids()
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        paths_used_ids = used_ids[paths[:, 0]]
        assert any((paths[i, -1] not in paths_used_ids[i] and paths[i, -1] != paths[i, 0]) for i in range(len(paths)))

    def test_kg_generate_path_metapaths_no_collaborative_no_temporal(self):
        metapaths = [[("item_id", "ra", "entity_id"), ("entity_id", "ra", "item_id")], ["rb", "rc"]]
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "metapath",
                "collaborative_path": False,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        # Differently from igraph, reverse paths must be explicitly defined in the KG
        potential_paths = np.array(
            [
                [2, 5, 6, 1, 11, 1, 7],  # ub->[UI-Relation]->eb->ra->ei->ra->ec
                [2, 5, 7, 1, 11, 1, 6],  # reverse of above ub->[UI-Relation]->ec->ra->ei->ra->eb
                [3, 5, 7, 1, 12, 1, 8],  # uc->[UI-Relation]->ec->ra->ej->ra->ed
                [2, 5, 6, 2, 13, 3, 7],  # ub->[UI-Relation]->eb->rb->ek->rc->ec
                [3, 5, 7, 2, 14, 3, 8],  # uc->[UI-Relation]->ec->rb->el->rc->ed
            ]
        )
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_metapaths_collaborative_no_temporal(self):
        metapaths = [
            [
                ("item_id", "[UI-Relation]", "user_id"),
                ("user_id", "[UI-Relation]", "item_id"),
                ("item_id", "ra", "entity_id"),
                ("entity_id", "ra", "item_id"),
            ],
            [("item_id", "[UI-Relation]", "user_id"), ("user_id", "[UI-Relation]", "item_id")],
        ]
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "metapath",
                "collaborative_path": True,
                "temporal_causality": False,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user_num = train_dataset.user_num
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_metapaths_no_collaborative_temporal(self):
        metapaths = [[("item_id", "ra", "entity_id"), ("entity_id", "ra", "item_id")], ["rb", "rc"]]
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "MAX_PATHS_PER_USER": 2,
            "path_sample_args": {
                "strategy": "metapath",
                "collaborative_path": False,
                "temporal_causality": True,
                "restrict_by_phase": True,
            },
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        user = train_dataset.inter_feat[train_dataset.uid_field].numpy()
        item = train_dataset.inter_feat[train_dataset.iid_field].numpy()
        timestamp = train_dataset.inter_feat[train_dataset.time_field].numpy()
        temporal_matrix = np.zeros((train_dataset.user_num, train_dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        paths = train_dataset.generate_user_paths()  # path ids not remapped
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - train_dataset.user_num
        subsequent_pos_items = paths[:, -1] - train_dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_tokenize_path_dataset(self):
        config_dict = {
            "model": "PEARLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "path_sample_args": {"path_token_separator": " "},
            "tokenizer": {
                "model": "WordLevel",
                "context_length": 24,
                "special_tokens": {
                    "mask_token": "[MASK]",
                    "unk_token": "[UNK]",
                    "pad_token": "[PAD]",
                    "eos_token": "[EOS]",
                    "bos_token": "[BOS]",
                    "sep_token": "[SEP]",
                    "cls_token": "[CLS]",
                },
                "template": "{bos_token}:0 $A:0 {eos_token}",
            },
        }
        dataset = new_dataset(config_dict=config_dict)
        path_string = "U2 R5 I2 R1 E7 R1 I3\nU2 R5 I3 R1 E7 R1 I2\nU2 R5 I2 R2 E9 R3 I3"
        dataset._path_dataset = path_string
        train_dataset, valid_dataset, test_dataset = dataset.build()
        tokenized_path_string = train_dataset.tokenized_dataset["input_ids"].tolist()
        split_path_string = path_string.split("\n")
        assert tokenized_path_string[0] == train_dataset.tokenizer(split_path_string[0])["input_ids"]
        assert tokenized_path_string[1] == train_dataset.tokenizer(split_path_string[1])["input_ids"]
        assert tokenized_path_string[2] == train_dataset.tokenizer(split_path_string[2])["input_ids"]


class TestKGGLMDataset(unittest.TestCase):
    def test_generate_pretrain_dataset(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "max_paths_per_user": 2,
            "path_sample_args": {"pretrain_paths": 5, "pretrain_hop_length": (1, 3)},
            "train_stage": "pretrain",
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_dataset, valid_dataset, test_dataset = split_dataset(config_dict=config_dict)
        paths = train_dataset.path_dataset.split("\n")
        path_lengths = [p.count("R") for p in paths]
        for p_len in range(
            config_dict["path_sample_args"]["pretrain_hop_length"][0],
            config_dict["path_sample_args"]["pretrain_hop_length"][1] + 1,
        ):
            assert p_len in path_lengths
        assert not any([PathLanguageModelingTokenType.USER.token in p for p in paths])


if __name__ == "__main__":
    pytest.main()
