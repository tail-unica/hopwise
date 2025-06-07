# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

import os

from transformers import GPT2LMHeadModel
from transformers.trainer_utils import get_last_checkpoint

from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM


class KGGLM(PEARLM):
    TRAIN_STAGES = ["lp_pretrain", "finetune"]

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

            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            self.hf_model = GPT2LMHeadModel.from_pretrained(self.pre_model_path)
