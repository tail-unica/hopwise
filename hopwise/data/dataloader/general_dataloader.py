# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""hopwise.data.dataloader.general_dataloader
################################################
"""

from logging import getLogger

import torch

from hopwise.data.dataloader.abstract_dataloader import AbstractDataLoader
from hopwise.data.interaction import Interaction


class TrainDataLoader(AbstractDataLoader):
    """TrainDataLoader is used for training. It can generate negative interaction when :attr:`training` is True.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    @property
    def pr_end(self):
        return len(self._dataset.inter_feat)

    def _shuffle(self):
        self._dataset.shuffle()

    def _next_batch_data(self):
        interaction = self._dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return interaction

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        interaction = self.sampler.sample_by_user_ids(interaction, self._dataset)
        return interaction


class FullSortRecEvalDataLoader(AbstractDataLoader):
    """FullSortRecEvalDataLoader is used for full sort evaluation. It can generate all candidate items for each user.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        
        # Get user dataframe for evaluation
        self.user_df = dataset.inter_feat.groupby(self.uid_field).first().reset_index()
        
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self.user_df)

    @property
    def pr_end(self):
        return len(self.user_df)

    def _shuffle(self):
        # No shuffling for evaluation
        pass

    def _next_batch_data(self):
        user_df = self.user_df[self.pr:self.pr + self.step]
        self.pr += self.step
        return user_df

    def collate_fn(self, index):
        index = torch.tensor(index)
        user_df = self.user_df.iloc[index]
        
        # Create interaction with all items for each user
        user_ids = user_df[self.uid_field].values
        n_users = len(user_ids)
        n_items = self._dataset.item_num
        
        # Create full interaction matrix
        user_ids_expanded = torch.tensor(user_ids).repeat_interleave(n_items)
        item_ids_expanded = torch.arange(n_items).repeat(n_users)
        
        interaction = Interaction({
            self.uid_field: user_ids_expanded,
            self.iid_field: item_ids_expanded
        })
        
        return interaction


class NegSampleDataLoader(AbstractDataLoader):
    """NegSampleDataLoader is used for negative sampling during training.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        interaction = self.sampler.sample_by_user_ids(interaction, self._dataset)
        return interaction


class FullSortLPEvalDataLoader(AbstractDataLoader):
    """FullSortLPEvalDataLoader is used for link prediction evaluation.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sampler = sampler
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self._batch_size = batch_size
        self.step = batch_size
        self.sample_size = len(self._dataset.inter_feat)

    def collate_fn(self, index):
        index = torch.tensor(index)
        interaction = self._dataset[index]
        return interaction
