# @Time   : 2022/7/15
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com
import os
import sys
import tempfile
import unittest

from hopwise.quick_start import objective_function
from hopwise.trainer import HyperTuning

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "test_hyper_tuning_config.yaml")]
params_file = os.path.join(current_path, "test_hyper_tuning_params.test")

# avoids distributed tuning (e.g., Ray) from cleaning up temporary directories (Python >= 3.10)
_TMPDIR_KWARGS = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}


def quick_test(tuner, algo):
    with tempfile.TemporaryDirectory(**_TMPDIR_KWARGS) as tmpdirname:
        if tuner == "ray":
            test_data_path = os.path.join(current_path, os.pardir, "test_data")
            ray_config_file = os.path.join(tmpdirname, "ray_data_path.yaml")
            with open(ray_config_file, "w") as f:
                f.write(f"data_path: {test_data_path}")

            test_config_file_list = [*config_file_list, ray_config_file]
        else:
            test_config_file_list = config_file_list

        hp = HyperTuning(
            objective_function,
            tuner=tuner,
            algo=algo,
            early_stop=10,
            max_evals=10,
            params_file=params_file,
            fixed_config_file_list=test_config_file_list,
            output_path=tmpdirname,
        )
        hp.run()


class TestHyperTuning(unittest.TestCase):
    def test_hyperopt_exhaustive(self):
        quick_test(tuner="hyperopt", algo="exhaustive")

    def test_hyperopt_random(self):
        quick_test(tuner="hyperopt", algo="random")

    def test_hyperopt_bayes(self):
        quick_test(tuner="hyperopt", algo="bayes")

    @unittest.skipIf(
        sys.version_info >= (3, 12),
        "hyperopt anneal is unsupported on Python >= 3.12 (numpy>=2 incompatibility; see HyperTuning.select_algo)",
    )
    def test_hyperopt_anneal(self):
        quick_test(tuner="hyperopt", algo="anneal")

    def test_ray_only_searcher(self):
        quick_test(tuner="ray", algo="hyperopt")

    def test_ray_only_scheduler(self):
        quick_test(tuner="ray", algo="medianstopping")

    def test_ray_searcher_and_scheduler(self):
        quick_test(tuner="ray", algo="random-async_hyperband")

    def test_optuna_only_sampler(self):
        quick_test(tuner="optuna", algo="TPESampler")

    def test_optuna_only_pruner(self):
        quick_test(tuner="optuna", algo="HyperbandPruner")

    def test_optuna_sampler_and_pruner(self):
        quick_test(tuner="optuna", algo="RandomSampler-MedianPruner")

    def test_optuna_grid_sampler(self):
        quick_test(tuner="optuna", algo="GridSampler")


if __name__ == "__main__":
    unittest.main()
