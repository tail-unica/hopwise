# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/9, 2020/9/29, 2021/7/15, 2022/7/6
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan, Gaowei Zhang
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, zgw15630559577@163.com  # noqa: E501

"""hopwise.data.dataloader.general_dataloader
################################################
"""

from logging import getLogger

import numpy as np
import torch

from hopwise.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from hopwise.data.interaction import Interaction, cat_interactions
from hopwise.utils import InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"])
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"
        self._set_neg_sample_args(config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"])
        if self.neg_sample_args["distribution"] != "none" and self.neg_sample_args["sample_num"] != "none":
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
            self.sample_size = len(self.uid_list)
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if self.neg_sample_args["distribution"] != "none" and self.neg_sample_args["sample_num"] != "none":
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if self.neg_sample_args["distribution"] != "none" and self.neg_sample_args["sample_num"] != "none":
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat((positive_i, self._dataset[index][self.iid_field]), 0)

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return the data samples with positives, not negatives

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()

        if shuffle:
            self.logger.warning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def check_sequential(self, config):
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL

    def _build_positive_samples(self, dataset, sampler, feat, source_field, target_field):
        source_num = len(dataset.field2id_token[source_field])

        self._source_list = []
        self._sample2positive_num = np.zeros(source_num, dtype=np.int64)
        self._sample2positives = np.array([None] * source_num)
        self._sample2history = np.array([None] * source_num)

        dataset.sort(by=source_field, ascending=True)
        last_source = None
        positives = set()
        used_ids = sampler.used_ids

        for source, target in zip(feat[source_field].numpy(), feat[target_field].numpy()):
            if source != last_source:
                self._set_source_property(last_source, used_ids[last_source], positives)
                last_source = source
                self._source_list.append(source)
                positives = set()
            positives.add(target)
        self._set_source_property(last_source, used_ids[last_source], positives)
        self._source_list = torch.tensor(self._source_list, dtype=torch.int64)
        self._source_df = dataset.join(Interaction({source_field: self._source_list}))
        self.sample_size = len(self._source_df) if not self.is_sequential else len(dataset)

    def _set_source_property(self, source, used_ids, positives):
        if source is None:
            return
        history = used_ids - positives
        self._sample2positives[source] = torch.tensor(list(positives), dtype=torch.int64)
        self._sample2positive_num[source] = len(positives)
        self._sample2history[source] = torch.tensor(list(history), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self._dataset.item_num, 1)
            new_batch_size = batch_num * self._dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def _not_sequential_collate_fn(self, index, source_field):
        index = np.array(index)
        source_df = self._source_df[index]
        source_list = list(source_df[source_field])

        history = self._sample2history[source_list]
        positives = self._sample2positives[source_list]

        history_source = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history)])
        history_target = torch.cat(list(history))

        positive_source = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positives)])
        positive_target = torch.cat(list(positives))

        return source_df, (history_source, history_target), positive_source, positive_target


class FullSortRecEvalDataLoader(FullSortEvalDataLoader):
    """:class:`FullSortRecEvalDataLoader` is a dataloader for full-sort evaluation for the recommendation (Rec) task.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.check_sequential(config)

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        self._build_positive_samples(dataset, sampler, dataset.inter_feat, self.uid_field, self.iid_field)
        self.uid2items_num = self._sample2positive_num
        self.uid2positive_item = self._sample2positives
        self.uid2history_item = self._sample2history
        self.uid_list = self._source_list
        self.user_df = self._source_df

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            return self._not_sequential_collate_fn(index, self.uid_field)
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i


class FullSortLPEvalDataLoader(FullSortEvalDataLoader):
    """:class:`FullSortLPEvalDataLoader` is a dataloader for full-sort evaluation for the link prediction (LP) task.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.check_sequential(config)

        self.head_entity_field = dataset.head_entity_field
        self.relation_field = dataset.relation_field
        self.tail_entity_field = dataset.tail_entity_field

        self._build_positive_samples(dataset, sampler, dataset.kg_feat, self.head_entity_field, self.tail_entity_field)
        self.head2tails_num = self._sample2positive_num
        self.head2positive_tail = self._sample2positives
        self.head2history_tail = self._sample2history
        self.head_list = self._source_list
        self.kg_df = self._source_df

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _build_positive_samples(self, dataset, sampler, feat, source_field, target_field):
        relation_field = dataset.relation_field
        head_list = list()
        relation_list = list()
        tail_list = list()
        self.source_num = len(dataset.field2id_token[source_field])

        self._sample2positive_num = np.zeros(self.source_num, dtype=np.int64)
        self._sample2positives = np.array([None] * self.source_num)
        self._sample2history = np.array([None] * self.source_num)

        last_source = None
        positives = set()
        used_ids = sampler.get_used_ids()

        for source, relation, target in zip(
            feat[source_field].numpy(), feat[relation_field], feat[target_field].numpy()
        ):
            if source != last_source:
                self._set_source_property(last_source, used_ids[last_source], positives)
                last_source = source
                head_list.append(source)
                relation_list.append(relation)
                tail_list.append(target)
                positives = set()
            positives.add(target)
        self._set_source_property(last_source, used_ids[last_source], positives)
        self._source_list = torch.tensor(head_list, dtype=torch.int64)
        tail_list = torch.tensor(tail_list, dtype=torch.int64)
        relation_list = torch.tensor(relation_list, dtype=torch.int64)
        self._source_df = dataset.join(
            Interaction({source_field: self._source_list, relation_field: relation_list, target_field: tail_list})
        )
        self.sample_size = len(self._source_df) if not self.is_sequential else len(dataset)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            return self._not_sequential_collate_fn(index, self.head_entity_field)
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i
