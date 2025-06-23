# @Time   : 2025/06
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

r"""KGGLM
##################################################
Reference:
    Balloccu et al. "KGGLM: A Generative Language Model for Generalizable Knowledge
    Graph Representation Learning in Recommendation." in RecSys 2024.

Reference code:
    https://github.com/mirkomarras/kgglm
"""

import os

from transformers.trainer_utils import get_last_checkpoint

from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM


class KGGLM(PEARLM):
    TRAIN_STAGES = ["pretrain", "finetune"]

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config["train_stage"]
        self.pre_model_path = config["pre_model_path"]

        assert self.train_stage in self.TRAIN_STAGES
        if self.train_stage == "finetune":
            # load pretrained model for finetune
            if not os.path.exists(os.path.join(self.pre_model_path, "config.json")):
                # if the path is not a checkpoint, we assume it contains the checkpoint
                self.pre_model_path = get_last_checkpoint(self.pre_model_path)

            from safetensors.torch import load_file

            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            weights = load_file(os.path.join(self.pre_model_path, "model.safetensors"))
            self.load_state_dict(weights, strict=False)
