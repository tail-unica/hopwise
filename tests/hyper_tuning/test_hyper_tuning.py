# @Time   : 2022/7/15
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com
import os
import tempfile
import unittest

from hopwise.quick_start import objective_function
from hopwise.trainer import HyperTuning

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "test_hyper_tuning_config.yaml")]
params_file = os.path.join(current_path, "test_hyper_tuning_params.test")


def quick_test(tuner, algo):
    with tempfile.TemporaryDirectory() as tmpdirname:
        if tuner == "ray":
            test_data_path = os.path.join(current_path, os.pardir, "test_data")
            ray_config_file = tempfile.NamedTemporaryFile(mode="w", delete=True)
            ray_config_file.write(f"data_path: {test_data_path}")
            ray_config_file.flush()

            test_config_file_list = [*config_file_list, ray_config_file.name]
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

        if tuner == "ray":
            ray_config_file.close()


class TestHyperTuning(unittest.TestCase):
    def test_hyperopt_exhaustive(self):
        quick_test(tuner="hyperopt", algo="exhaustive")

    def test_hyperopt_random(self):
        quick_test(tuner="hyperopt", algo="random")

    def test_hyperopt_bayes(self):
        quick_test(tuner="hyperopt", algo="bayes")

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
