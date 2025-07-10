# @Time    :   2020/11/1
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE:
# @Time   : 2021/7/1
# @Author : Xingyu Pan
# @Email  : xy_pan@foxmail.com

import logging
import os
import tempfile
import unittest
import warnings

from hopwise.quick_start import run_hopwise


def run_params(parm_dict, extra_dict=None):
    with tempfile.TemporaryDirectory() as tempdir:
        config_dict = {"epochs": 1, "state": "INFO", "checkpoint_dir": tempdir}
        for name, parms in parm_dict.items():
            for parm in parms:
                config_dict[name] = parm
                if extra_dict is not None:
                    config_dict.update(extra_dict)
                try:
                    run_hopwise(model="BPR", dataset="ml-100k", config_dict=config_dict)
                except Exception:
                    print(f"\ntest `{name}`={parm} ... fail.\n")
                    logging.critical(f"\ntest `{name}`={parm} ... fail.\n", exc_info=True)

                    return False
    return True


class TestOverallConfig(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def test_gpu_id(self):
        self.assertTrue(run_params({"gpu_id": ["0", "-1", "1"]}))

    def test_use_gpu(self):
        self.assertTrue(run_params({"use_gpu": [True, False]}))

    def test_reproducibility(self):
        self.assertTrue(run_params({"reproducibility": [True, False]}))

    def test_seed(self):
        self.assertTrue(run_params({"seed": [2021, 1024]}))

    def test_data_path(self):
        self.assertTrue(run_params({"data_path": ["dataset/", "./dataset"]}))

    def test_epochs(self):
        self.assertTrue(run_params({"epochs": [0, 1, 2]}))

    def test_train_batch_size(self):
        self.assertTrue(run_params({"train_batch_size": [1, 2048, 200000]}))

    def test_learner(self):
        self.assertTrue(run_params({"learner": ["adam", "sgd", "foo"]}))

    def test_learning_rate(self):
        self.assertTrue(run_params({"learning_rate": [0, 0.001, 1e-5]}))

    def test_training_neg_sampling(self):
        self.assertTrue(
            run_params(
                {
                    "train_neg_sample_args": [
                        {"distribution": "uniform", "sample_num": 1},
                        {"distribution": "uniform", "sample_num": 2},
                        {"distribution": "uniform", "sample_num": 3},
                    ]
                }
            )
        )

    def test_transform(self):
        self.assertTrue(run_params({"transform": [None]}))

    def test_eval_step(self):
        self.assertTrue(run_params({"eval_step": [1, 2]}))

    def test_stopping_step(self):
        self.assertTrue(run_params({"stopping_step": [0, 1, 2]}))

    def test_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.assertTrue(
                run_params({"checkpoint_dir": [os.path.join(tempdir, "saved_1"), os.path.join(tempdir, "saved_2")]})
            )

    def test_eval_batch_size(self):
        self.assertTrue(run_params({"eval_batch_size": [1, 100]}))

    def test_topk(self):
        settings = {
            "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
            "valid_metric": "Recall@1",
        }
        self.assertTrue(run_params({"topk": [1, [1, 3]]}, extra_dict=settings))

    def test_loss(self):
        settings = {
            "metrics": ["MAE", "RMSE", "LOGLOSS", "AUC"],
            "valid_metric": "auc",
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "RO",
                "mode": "uni100",
            },
        }
        self.assertTrue(
            run_params(
                {
                    "topk": {
                        1,
                    }
                },
                extra_dict=settings,
            )
        )

    def test_metric(self):
        settings = {"topk": 3, "valid_metric": "Recall@3"}
        self.assertTrue(
            run_params(
                {"metrics": ["Recall", ["Recall", "MRR", "NDCG", "Hit", "Precision"]]},
                extra_dict=settings,
            )
        )

    def test_split_ratio(self):
        self.assertTrue(
            run_params(
                {
                    "eval_args": [
                        {"split": {"RS": [0.8, 0.1, 0.1]}},
                        {"split": {"RS": [16, 2, 2]}},
                    ]
                }
            )
        )

    def test_group_by_user(self):
        self.assertTrue(run_params({"eval_args": [{"group_by": "user"}, {"group_by": "None"}]}))

    def test_use_mixed_precision(self):
        self.assertTrue(run_params({"enable_amp": [True, False]}))

    def test_use_grad_scaler(self):
        self.assertTrue(run_params({"enable_scaler": [True, False]}))

    def test_progress_bar_rich(self):
        self.assertTrue(run_params({"progress_bar_rich": [True, False]}))


if __name__ == "__main__":
    unittest.main(verbosity=1)
