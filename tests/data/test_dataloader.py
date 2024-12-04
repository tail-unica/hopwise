# @Time   : 2021/1/5
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time    :   2020/1/5, 2021/7/1, 2021/7/19
# @Author  :   Yushuo Chen, Xingyu Pan, Zhichao Feng
# @email   :   chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, fzcbupt@gmail.com

import logging
import os

import numpy as np
import pytest

from hopwise.config import Config
from hopwise.data import create_dataset, data_preparation
from hopwise.data.dataloader.general_dataloader import (
    FullSortEvalDataLoader,
    NegSampleEvalDataLoader,
)
from hopwise.utils import init_seed

current_path = os.path.dirname(os.path.realpath(__file__))


def new_dataloader(config_dict=None, config_file_list=None):
    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    return data_preparation(config, dataset)


class TestGeneralDataloader:
    def test_general_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "labeled",
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_dataloader(data, item_list, batch_size, train=False):
            data.shuffle = False
            pr = 0
            for batch_data in data:
                batch_item_list = item_list[pr : pr + batch_size]
                if train:
                    user_df = batch_data
                else:
                    user_df = batch_data[0]
                assert (user_df["item_id"].numpy() == batch_item_list).all()
                pr += batch_size

        check_dataloader(train_data, list(range(1, 41)), train_batch_size, True)
        check_dataloader(valid_data, list(range(41, 46)), eval_batch_size)
        check_dataloader(test_data, list(range(46, 51)), eval_batch_size)

    def test_general_neg_sample_dataloader_in_pair_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            batch_item_list = train_item_list[pr : pr + train_batch_size]
            assert (batch_data["item_id"].numpy() == batch_item_list).all()
            assert (batch_data["item_id"] == batch_data["price"]).all()
            assert (40 < batch_data["neg_item_id"]).all()
            assert (batch_data["neg_item_id"] <= 100).all()
            assert (batch_data["neg_item_id"] == batch_data["neg_price"]).all()
            pr += train_batch_size

    def test_general_neg_sample_dataloader_in_point_wise(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "DMF",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        train_data.shuffle = False
        train_item_list = list(range(1, 41))
        pr = 0
        for batch_data in train_data:
            step = len(batch_data) // 2
            batch_item_list = train_item_list[pr : pr + step]
            assert (batch_data["item_id"][:step].numpy() == batch_item_list).all()
            assert (40 < batch_data["item_id"][step:]).all()
            assert (batch_data["item_id"][step:] <= 100).all()
            assert (batch_data["item_id"] == batch_data["price"]).all()
            pr += step

    def test_general_full_dataloader(self):
        train_batch_size = 6
        eval_batch_size = 100
        config_dict = {
            "model": "BPR",
            "dataset": "general_full_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "full",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, history_index, positive_u, positive_i = batch_data
                history_row, history_col = history_index
                assert len(user_df) == result[i]["len_user_df"]
                assert (user_df["user_id"].numpy() == result[i]["user_df_user_id"]).all()
                assert len(history_row) == len(history_col) == result[i]["history_len"]
                assert (history_row.numpy() == result[i]["history_row"]).all()
                assert (history_col.numpy() == result[i]["history_col"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "len_user_df": 1,
                "user_df_user_id": [1],
                "history_len": 40,
                "history_row": 0,
                "history_col": list(range(1, 41)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [41, 42, 43, 44, 45],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [2],
                "history_len": 37,
                "history_row": 0,
                "history_col": list(range(1, 38)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [38, 39, 40, 41, 42],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [3],
                "history_len": 0,
                "history_row": [],
                "history_col": [],
                "positive_u": [0],
                "positive_i": [1],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "len_user_df": 1,
                "user_df_user_id": [1],
                "history_len": 45,
                "history_row": 0,
                "history_col": list(range(1, 46)),
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [46, 47, 48, 49, 50],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [2],
                "history_len": 37,
                "history_row": 0,
                "history_col": list(range(1, 36)) + [41, 42],
                "positive_u": [0, 0, 0, 0, 0],
                "positive_i": [36, 37, 38, 39, 40],
            },
            {
                "len_user_df": 1,
                "user_df_user_id": [3],
                "history_len": 0,
                "history_row": [],
                "history_col": [],
                "positive_u": [0],
                "positive_i": [1],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_101(self):
        train_batch_size = 6
        eval_batch_size = 101
        config_dict = {
            "model": "BPR",
            "dataset": "general_uni100_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "uni100",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data._batch_size == 202
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, row_idx, positive_u, positive_i = batch_data
                assert result[i]["item_id_check"](user_df["item_id"])
                assert (row_idx.numpy() == result[i]["row_idx"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "item_id_check": lambda data: data[0] == 9 and (8 < data[1:]).all() and (data[1:] <= 100).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [9],
            },
            {
                "item_id_check": lambda data: data[0] == 1 and (data[1:] != 1).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [17, 18]).all()
                and (16 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [17, 18],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "item_id_check": lambda data: data[0] == 10 and (9 < data[1:]).all() and (data[1:] <= 100).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [10],
            },
            {
                "item_id_check": lambda data: data[0] == 1 and (data[1:] != 1).all(),
                "row_idx": [0] * 101,
                "positive_u": [0],
                "positive_i": [1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [19, 20]).all()
                and (18 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [19, 20],
            },
        ]
        check_result(test_data, test_result)

    def test_general_uni100_dataloader_with_batch_size_in_303(self):
        train_batch_size = 6
        eval_batch_size = 303
        config_dict = {
            "model": "BPR",
            "dataset": "general_uni100_dataloader",
            "data_path": current_path,
            "load_col": None,
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": "uni100",
            },
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)

        def check_result(data, result):
            assert data._batch_size == 303
            assert len(data) == len(result)
            for i, batch_data in enumerate(data):
                user_df, row_idx, positive_u, positive_i = batch_data
                assert result[i]["item_id_check"](user_df["item_id"])
                assert (row_idx.numpy() == result[i]["row_idx"]).all()
                assert (positive_u.numpy() == result[i]["positive_u"]).all()
                assert (positive_i.numpy() == result[i]["positive_i"]).all()

        valid_result = [
            {
                "item_id_check": lambda data: data[0] == 9
                and (8 < data[1:101]).all()
                and (data[1:101] <= 100).all()
                and data[101] == 1
                and (data[102:202] != 1).all(),
                "row_idx": [0] * 101 + [1] * 101,
                "positive_u": [0, 1],
                "positive_i": [9, 1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [17, 18]).all()
                and (16 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [17, 18],
            },
        ]
        check_result(valid_data, valid_result)

        test_result = [
            {
                "item_id_check": lambda data: data[0] == 10
                and (9 < data[1:101]).all()
                and (data[1:101] <= 100).all()
                and data[101] == 1
                and (data[102:202] != 1).all(),
                "row_idx": [0] * 101 + [1] * 101,
                "positive_u": [0, 1],
                "positive_i": [10, 1],
            },
            {
                "item_id_check": lambda data: (data[0:2].numpy() == [19, 20]).all()
                and (18 < data[2:]).all()
                and (data[2:] <= 100).all(),
                "row_idx": [0] * 202,
                "positive_u": [0, 0],
                "positive_i": [19, 20],
            },
        ]
        check_result(test_data, test_result)

    def test_general_diff_dataloaders_in_valid_test_phases(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": {"valid": "uni100", "test": "full"},
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        assert isinstance(valid_data, NegSampleEvalDataLoader)
        assert isinstance(test_data, FullSortEvalDataLoader)

    def test_general_diff_eval_neg_sample_args_in_valid_test_phases(self):
        train_batch_size = 6
        eval_batch_size = 2
        config_dict = {
            "model": "BPR",
            "dataset": "general_dataloader",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "mode": {"valid": "uni100", "test": "pop200"},
            },
            "train_neg_sample_args": None,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle": False,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        assert isinstance(valid_data, NegSampleEvalDataLoader)
        assert isinstance(test_data, NegSampleEvalDataLoader)
        assert valid_data.neg_sample_args["distribution"] == "uniform"
        assert valid_data.neg_sample_args["sample_num"] == 100
        assert test_data.neg_sample_args["distribution"] == "popularity"
        assert test_data.neg_sample_args["sample_num"] == 200


class TestKnowledgePathDataLoader:
    """
    Expected paths:
    ea->rd->eb
    """

    def test_kg_generate_path_weighted_random_walk_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "weighted-rw",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": False,
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
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        used_ids = train_data.general_dataloader._sampler.used_ids
        # randomness forces us to retry until we get a valid path
        while True:
            paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
            if len(paths) > 0:
                break
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_weighted_random_walk_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "weighted-rw",
            "max_paths_per_user": 2,
            "collaborative_path": True,
            "temporal_causality": False,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        user_num = train_data.dataset.user_num
        used_ids = train_data.general_dataloader._sampler.used_ids
        # randomness forces us to retry until we get a valid path
        while True:
            paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
            if len(paths) > 0:
                break
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_weighted_random_walk_no_collaborative_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "weighted-rw",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": True,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        dataset = train_data.dataset
        user = dataset.inter_feat[dataset.uid_field].numpy()
        item = dataset.inter_feat[dataset.iid_field].numpy()
        timestamp = dataset.inter_feat[dataset.time_field].numpy()
        temporal_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        used_ids = train_data.general_dataloader._sampler.used_ids
        # randomness forces us to retry until we get a valid path
        while True:
            paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
            if len(paths) > 0:
                break
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - dataset.user_num
        subsequent_pos_items = paths[:, -1] - dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_generate_path_constrained_random_walk_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "constrained-rw",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": False,
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
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_constrained_random_walk_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "constrained-rw",
            "max_paths_per_user": 2,
            "collaborative_path": True,
            "temporal_causality": False,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        user_num = train_data.dataset.user_num
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_constrained_random_walk_no_collaborative_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "constrained-rw",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": True,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        dataset = train_data.dataset
        user = dataset.inter_feat[dataset.uid_field].numpy()
        item = dataset.inter_feat[dataset.iid_field].numpy()
        timestamp = dataset.inter_feat[dataset.time_field].numpy()
        temporal_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - dataset.user_num
        subsequent_pos_items = paths[:, -1] - dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_generate_path_all_simple_no_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "simple",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": False,
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
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        potential_paths_found = paths[:, None] == potential_paths
        assert potential_paths_found.all(axis=-1).any()

    def test_kg_generate_path_all_simple_collaborative_no_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "simple",
            "max_paths_per_user": 2,
            "collaborative_path": True,
            "temporal_causality": False,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        user_num = train_data.dataset.user_num
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_all_simple_no_collaborative_temporal(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "simple",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": True,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        dataset = train_data.dataset
        user = dataset.inter_feat[dataset.uid_field].numpy()
        item = dataset.inter_feat[dataset.iid_field].numpy()
        timestamp = dataset.inter_feat[dataset.time_field].numpy()
        temporal_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - dataset.user_num
        subsequent_pos_items = paths[:, -1] - dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_generate_path_metapaths_no_collaborative_no_temporal(self):
        metapaths = [[("item_id", "ra", "entity_id"), ("entity_id", "ra", "item_id")], ["rb", "rc"]]
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "metapath",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": False,
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
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
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
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "metapath",
            "max_paths_per_user": 2,
            "collaborative_path": True,
            "temporal_causality": False,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        user_num = train_data.dataset.user_num
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        assert (paths[:, 4] < user_num).any()

    def test_kg_generate_path_metapaths_no_collaborative_temporal(self):
        metapaths = [[("item_id", "ra", "entity_id"), ("entity_id", "ra", "item_id")], ["rb", "rc"]]
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "path_sampling_strategy": "metapath",
            "max_paths_per_user": 2,
            "collaborative_path": False,
            "temporal_causality": True,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "metapaths": metapaths,
        }
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        dataset = train_data.dataset
        user = dataset.inter_feat[dataset.uid_field].numpy()
        item = dataset.inter_feat[dataset.iid_field].numpy()
        timestamp = dataset.inter_feat[dataset.time_field].numpy()
        temporal_matrix = np.zeros((dataset.user_num, dataset.item_num), dtype=timestamp.dtype)
        temporal_matrix[user, item] = timestamp
        used_ids = train_data.general_dataloader._sampler.used_ids
        paths = train_data.dataset.generate_paths(used_ids)  # path ids not remapped
        users = paths[:, 0]
        starting_pos_items = paths[:, 2] - dataset.user_num
        subsequent_pos_items = paths[:, -1] - dataset.user_num
        assert (temporal_matrix[users, starting_pos_items] < temporal_matrix[users, subsequent_pos_items]).all()

    def test_kg_add_paths_relations(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "path_hop_length": 3,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
        }
        paths = np.array(
            [
                [2, 6, 11, 7],  # ub->eb->ei->ec
                [2, 7, 11, 6],  # reverse of above
                [2, 6, 13, 7],  # ub->eb->ek->ec
                # [3, 7, 14, -1],  # uc->ec->-el->-1  # not supported yet
            ]
        )
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        graph = train_data.dataset._create_ckg_igraph(show_relation=True, directed=False)
        paths_with_relations = train_data.dataset._add_paths_relations(graph, paths)
        assert paths_with_relations[0].tolist() == [2, 5, 6, 1, 11, 1, 7]
        assert paths_with_relations[1].tolist() == [2, 5, 7, 1, 11, 1, 6]
        assert paths_with_relations[2].tolist() == [2, 5, 6, 2, 13, 3, 7]
        # assert paths_with_relations[3] == [3, 5, 7, 2, 14, -1, -1]  # not supported yet

    def test_kg_tokenize_path_dataset(self):
        config_dict = {
            "model": "KGGLM",
            "dataset": "kg_generate_path",
            "data_path": current_path,
            "load_col": None,
            "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO"},
            "reasoning_path_template": "{user} {pos_iid} {entity_list} {rec_iid}",
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
        train_data, valid_data, test_data = new_dataloader(config_dict=config_dict)
        path_string = "U2 R5 I2 R1 E7 R1 I3\nU2 R5 I3 R1 E7 R1 I2\nU2 R5 I2 R2 E9 R3 I3"
        train_data.dataset._path_dataset = path_string
        tokenized_dataset = train_data.dataset.tokenize_path_dataset(phase="train")
        tokenized_path_string = tokenized_dataset.data["train"]["input_ids"].to_pylist()
        assert tokenized_path_string[0] == [4, 27, 24, 14, 20, 1, 20, 15, 3]
        assert tokenized_path_string[1] == [4, 27, 24, 15, 20, 1, 20, 14, 3]
        assert tokenized_path_string[2] == [4, 27, 24, 14, 21, 11, 22, 15, 3]


if __name__ == "__main__":
    pytest.main()
