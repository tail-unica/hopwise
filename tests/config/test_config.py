# @Time   : 2020/10/19
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/1
# @Author : Xingyu Pan
# @Email  : xy_pan@foxmail.com

import os
import sys
import unittest

from hopwise.config import Config

parameters_dict = {
    "model": "SASRec",
    "learning_rate": 0.2,
    "topk": [50, 100],
    "epochs": 100,
}

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "test_config_example.yaml")]


class TestConfigClass(unittest.TestCase):
    def test_default_settings(self):
        config = Config(model="BPR", dataset="ml-100k")

        self.assertEqual(config["model"], "BPR")
        self.assertEqual(config["dataset"], "ml-100k")

        self.assertIsInstance(config["gpu_id"], str)
        self.assertIsInstance(config["worker"], int)
        self.assertIsInstance(config["seed"], int)
        self.assertIsInstance(config["state"], str)
        self.assertIsInstance(config["data_path"], str)

        self.assertIsInstance(config["epochs"], int)
        self.assertIsInstance(config["train_batch_size"], int)
        self.assertIsInstance(config["learner"], str)
        self.assertIsInstance(config["learning_rate"], float)
        self.assertIsInstance(config["train_neg_sample_args"], dict)
        self.assertIsInstance(config["eval_step"], int)
        self.assertIsInstance(config["stopping_step"], int)
        self.assertIsInstance(config["checkpoint_dir"], str)

        self.assertIsInstance(config["eval_args"], dict)
        self.assertIsInstance(config["metrics"], list)
        self.assertIsInstance(config["topk"], list)
        self.assertIsInstance(config["valid_metric"], str)
        self.assertIsInstance(config["eval_batch_size"], int)

    def test_default_context_settings(self):
        config = Config(model="FM", dataset="ml-100k")

        self.assertEqual(config["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertEqual(config["eval_args"]["order"], "RO")
        self.assertEqual(config["eval_args"]["mode"]["valid"], "labeled")
        self.assertEqual(config["eval_args"]["mode"]["test"], "labeled")
        self.assertEqual(config["eval_args"]["group_by"], None)

        self.assertEqual(config["metrics"], ["AUC", "LogLoss"])
        self.assertEqual(config["valid_metric"], "AUC")
        self.assertEqual(
            config["train_neg_sample_args"],
            {
                "distribution": "none",
                "sample_num": "none",
                "alpha": "none",
                "dynamic": False,
                "candidate_num": 0,
            },
        )

    def test_default_sequential_settings(self):
        para_dict = {"train_neg_sample_args": None}
        config = Config(model="SASRec", dataset="ml-100k", config_dict=para_dict)
        self.assertEqual(config["eval_args"]["split"], {"LS": "valid_and_test"})
        self.assertEqual(config["eval_args"]["order"], "TO")
        self.assertEqual(config["eval_args"]["mode"]["valid"], "full")
        self.assertEqual(config["eval_args"]["mode"]["test"], "full")
        self.assertEqual(config["eval_args"]["group_by"], "user")

    def test_config_file_list(self):
        config = Config(model="BPR", dataset="ml-100k", config_file_list=config_file_list)

        self.assertEqual(config["model"], "BPR")
        self.assertEqual(config["learning_rate"], 0.1)
        self.assertEqual(config["topk"], [5, 20])
        self.assertEqual(config["eval_args"]["split"], {"LS": "valid_and_test"})
        self.assertEqual(config["eval_args"]["order"], "TO")
        self.assertEqual(config["eval_args"]["mode"]["valid"], "uni100")
        self.assertEqual(config["eval_args"]["mode"]["test"], "full")
        self.assertEqual(config["eval_args"]["group_by"], "user")

    def test_config_dict(self):
        config = Config(model="BPR", dataset="ml-100k", config_dict=parameters_dict)

        self.assertEqual(config["model"], "BPR")
        self.assertEqual(config["learning_rate"], 0.2)
        self.assertEqual(config["topk"], [50, 100])
        self.assertEqual(config["eval_args"]["split"], {"RS": [0.8, 0.1, 0.1]})
        self.assertEqual(config["eval_args"]["order"], "RO")
        self.assertEqual(config["eval_args"]["mode"]["valid"], "full")
        self.assertEqual(config["eval_args"]["mode"]["test"], "full")
        self.assertEqual(config["eval_args"]["group_by"], "user")

    def test_priority(self):
        config = Config(
            model="BPR",
            dataset="ml-100k",
            config_file_list=config_file_list,
            config_dict=parameters_dict,
        )

        self.assertEqual(config["learning_rate"], 0.2)  # default, file, dict
        self.assertEqual(config["topk"], [50, 100])  # default, file, dict
        self.assertEqual(config["eval_args"]["split"], {"LS": "valid_and_test"})
        self.assertEqual(config["eval_args"]["order"], "TO")
        self.assertEqual(config["eval_args"]["mode"]["valid"], "uni100")
        self.assertEqual(config["eval_args"]["mode"]["test"], "full")
        self.assertEqual(config["eval_args"]["group_by"], "user")
        self.assertEqual(config["epochs"], 100)  # default, dict

    def test_config_command_line(self):
        sys.argv.append("--use_gpu=False")
        sys.argv.append("--valid_metric=Recall@10")
        sys.argv.append('--metrics=["Recall"]')
        sys.argv.append("--epochs=200")
        sys.argv.append("--learning_rate=0.3")
        sys.argv.append("--train_neg_sample_args.sample_num=12")

        config = Config(model="BPR", dataset="ml-100k")

        self.assertEqual(config["use_gpu"], False)
        self.assertEqual(config["metrics"], ["Recall"])
        self.assertEqual(config["train_neg_sample_args"]["sample_num"], 12)
        self.assertEqual(config["learning_rate"], 0.3)
        self.assertEqual(config["epochs"], 200)


if __name__ == "__main__":
    unittest.main()
