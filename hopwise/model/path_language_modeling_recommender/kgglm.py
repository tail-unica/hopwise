from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM


class KGGLM(PEARLM):
    TRAIN_STAGES = ["lp_pretrain", "finetune"]

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config["train_stage"]
        self.pre_model_path = config["pre_model_path"]

        assert self.train_stage in self.TRAIN_STAGES
        if self.train_stage == "lp_pretrain":
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            from transformers import AutoModel

            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            self.model = AutoModel.from_pretrained(self.pre_model_path)
