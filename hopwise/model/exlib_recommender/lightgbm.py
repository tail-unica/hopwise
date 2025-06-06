# @Time   : 2020/1/17
# @Author : Chen Yang
# @Email  : 254170321@qq.com

r"""hopwise.model.exlib_recommender.lightgbm
##########################################
"""

import lightgbm as lgb

from hopwise.utils import InputType, ModelType


class LightGBM(lgb.Booster):
    r"""LightGBM is inherited from lgb.Booster"""

    type = ModelType.DECISIONTREE
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(lgb.Booster, self).__init__()

    def to(self, device):
        return self

    def load_state_dict(self, model_file):
        r"""Load state dictionary

        Args:
            model_file (str): file path of saved model

        """
        self = lgb.Booster(model_file=model_file)  # noqa: F841, PLW0642

    def load_other_parameter(self, other_parameter):
        r"""Load other parameters"""
        pass
