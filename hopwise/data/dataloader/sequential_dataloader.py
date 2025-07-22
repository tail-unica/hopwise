# UPDATE
# @Time   : 2025/07
# @Author : Alessandro
# @Email  : alessandro.soccol@unica.it


"""hopwise.data.dataloader.sequential_dataloader
################################################
"""

from logging import getLogger

import numpy as np

from hopwise.data.dataloader.general_dataloader import AbstractDataLoader
from hopwise.data.interaction import Interaction


class SequentialAugmentedDataloader(AbstractDataLoader):
    """:class:`SequentialAugmentedDataloader` is a dataloader for SequentialDataset.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return transformed_data


class SequentialDataloader(SequentialAugmentedDataloader):
    """:class:`SequentialDataloader` is a dataloader for training.

    Differently from SequentialAugmentedDataloader, it removes the data augmentation processing in SequentialDataset

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.max_seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_seq = self.create_item_seq(dataset)
        self.sample_size = len(self.item_seq)
        super(SequentialAugmentedDataloader, self).__init__(config, dataset, sampler, shuffle=shuffle)

    def create_item_seq(self, dataset):
        import pandas as pd
        import torch

        fields = [dataset.uid_field, dataset.iid_field]
        df = pd.DataFrame({field: dataset.inter_feat[field] for field in fields})

        item_seq, item_seq_len = list(), list()
        for user_id, group in df.groupby(dataset.uid_field):
            user_seq = group[dataset.iid_field].tolist()
            for i in range(0, len(user_seq), self.max_seq_len):
                chunk = user_seq[i : i + self.max_seq_len]
                item_seq_len.append(len(chunk))
                if len(chunk) < self.max_seq_len:
                    # pad item sequence if it's shorter than max_seq_len
                    item_seq.append(chunk + [0] * (self.max_seq_len - len(chunk)))
                else:
                    # otherwise, append the sequence
                    item_seq.append(chunk)

        return Interaction(
            {
                dataset.item_id_list_field: torch.tensor(item_seq),
                dataset.item_list_length_field: torch.tensor(item_seq_len),
            }
        )

    def collate_fn(self, index):
        index = np.array(index)
        return self.item_seq[index]
