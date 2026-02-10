# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/18, 2020/9/21, 2020/8/31
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Kaiyuan Li
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com

# UPDATE
# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it


"""hopwise.data.dataloader.knowledge_dataloader
################################################
"""

from logging import getLogger

import numpy as np

from hopwise.data.dataloader.abstract_dataloader import AbstractDataLoader
from hopwise.data.dataloader.general_dataloader import FullSortRecEvalDataLoader, TrainDataLoader
from hopwise.data.interaction import Interaction
from hopwise.utils import KGDataLoaderState, PathLanguageModelingTokenType, set_color


class KGDataLoader(AbstractDataLoader):
    """:class:`KGDataLoader` is a dataloader which would return the triplets with negative examples
    in a knowledge graph.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (KGSampler): The knowledge graph sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`KGDataLoader`, it's guaranteed to be ``True``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        if shuffle is False:
            shuffle = True
            self.logger.warning("kg based dataloader must shuffle the data")

        self.neg_sample_num = 1

        self.neg_prefix = config["NEG_PREFIX"]
        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field

        # kg negative cols
        self.neg_tid_field = self.neg_prefix + self.tid_field
        dataset.copy_field_property(self.neg_tid_field, self.tid_field)

        self.sample_size = len(dataset.kg_feat)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        cur_data = self._dataset.kg_feat[index]
        head_ids = cur_data[self.hid_field].numpy()
        neg_tail_ids = self._sampler.sample_by_entity_ids(head_ids, self.neg_sample_num)
        cur_data.update(Interaction({self.neg_tid_field: neg_tail_ids}))
        return cur_data


class KnowledgeBasedDataLoader:
    """:class:`KnowledgeBasedDataLoader` is used for knowledge based model.
    It has three states, which is saved in :attr:`state`.
    In different states, :meth:`~_next_batch_data` will return different :class:`~hopwise.data.interaction.Interaction`.
    Detailed, please see :attr:`~state`.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        kg_sampler (KGSampler): The knowledge graph sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        state (KGDataLoaderState):
            This dataloader has three states:
                - :obj:`~hopwise.utils.enum_type.KGDataLoaderState.RS`
                - :obj:`~hopwise.utils.enum_type.KGDataLoaderState.KG`
                - :obj:`~hopwise.utils.enum_type.KGDataLoaderState.RSKG`

            In the first state, this dataloader would only return the user-item interaction.
            In the second state, this dataloader would only return the triplets with negative
            examples in a knowledge graph.
            In the last state, this dataloader would return both knowledge graph information
            and user-item interaction information.
    """  # noqa: E501

    def __init__(self, config, dataset, sampler, kg_sampler, shuffle=False):
        self.config = config
        self.logger = getLogger()
        # using sampler
        self.general_dataloader = TrainDataLoader(config, dataset, sampler, shuffle=shuffle)

        # using kg_sampler
        self.kg_dataloader = KGDataLoader(config, dataset, kg_sampler, shuffle=True)

        self.shuffle = False
        self.state = None
        self.dataset = self._dataset = dataset
        self.kg_iter, self.gen_iter = None, None
        self._split = None

    def update_config(self, config):
        self.general_dataloader.update_config(config)
        self.kg_dataloader.update_config(config)

    def __iter__(self):
        if self.state is None:
            raise ValueError(
                "The dataloader's state must be set when using the kg based dataloader, "
                "you should call set_mode() before __iter__()"
            )
        if self.state == KGDataLoaderState.KG:
            return self.kg_dataloader.__iter__()
        elif self.state == KGDataLoaderState.RS:
            return self.general_dataloader.__iter__()
        elif self.state == KGDataLoaderState.RSKG:
            self.kg_iter = self.kg_dataloader.__iter__()
            self.gen_iter = self.general_dataloader.__iter__()
            return self

    def __next__(self):
        try:
            kg_data = next(self.kg_iter)
        except StopIteration:
            self.kg_iter = self.kg_dataloader.__iter__()
            kg_data = next(self.kg_iter)
        recdata = next(self.gen_iter)
        recdata.update(kg_data)
        return recdata

    def __len__(self):
        if self.state == KGDataLoaderState.KG:
            return len(self.kg_dataloader)
        else:
            return len(self.general_dataloader)

    def set_mode(self, state):
        """Set the mode of :class:`KnowledgeBasedDataLoader`, it can be set to three states:
            - KGDataLoaderState.RS
            - KGDataLoaderState.KG
            - KGDataLoaderState.RSKG

        The state of :class:`KnowledgeBasedDataLoader` would affect the result of _next_batch_data().

        Args:
            state (KGDataLoaderState): the state of :class:`KnowledgeBasedDataLoader`.
        """
        if state not in set(KGDataLoaderState):
            raise NotImplementedError(f"Kg data loader has no state named [{self.state}].")
        self.state = state

    def get_model(self, model):
        """Let the general_dataloader get the model, used for dynamic sampling."""
        self.general_dataloader.get_model(model)

    def knowledge_shuffle(self, epoch_seed):
        """Reset the seed to ensure that each subprocess generates the same index squence."""
        self.kg_dataloader.sampler.set_epoch(epoch_seed)

        if self.general_dataloader.shuffle:
            self.general_dataloader.sampler.set_epoch(epoch_seed)

    @property
    def split(self):
        return self._split

    def __str__(self):
        inter_feat = self._dataset.inter_feat
        kg_feat = self._dataset.kg_feat
        dataset = self.config["dataset"]
        benchmark_item_file = self.config["benchmark_item_filename"]

        info = [set_color(f"{self.split} {dataset}", "yellow")]
        info.extend(
            [
                set_color("The number of interactions", "blue") + f": {len(inter_feat)}",
                set_color("The number of users", "blue") + f": {len(inter_feat[self._dataset.uid_field].unique())}",
                set_color("Average actions of users in inters", "blue") + f": {self._dataset.avg_actions_of_users}",
                set_color("The number of items in inters", "blue") + f": {self._dataset.item_num}",
                set_color("Average actions of items in inters", "blue") + f": {self._dataset.avg_actions_of_items}",
                set_color("The sparsity of the dataset", "blue") + f": {self.sparsity_interaction_data_split * 100}%",
            ]
        )
        # data about knowledge split
        info.extend(
            [
                set_color("The number of triplets", "green") + f": {len(kg_feat)}",
                set_color("The number of head entities", "green") + f": {len(kg_feat.head_id.unique())}",
                set_color(f"Average actions of head entities in {self.split} knowledge graph", "green")
                + f": {self.avg_actions(kg_feat, 'head_id')}",
                set_color("The number of tail entities", "green") + f": {len(kg_feat.tail_id.unique())}",
                set_color(f"Average actions of tail entities in {self.split} knowledge graph", "green")
                + f": {self.avg_actions(kg_feat, 'tail_id')}",
                set_color("The sparsity of the Knowledge Graph", "green")
                + f": {self.sparsity_interaction_data_split * 100}%",
            ]
        )
        if benchmark_item_file is not None:
            import numpy as np

            item_feat = getattr(self._dataset, f"item_feat_{self.split}")[self._dataset.iid_field].to_numpy()
            info.extend(
                [
                    set_color(f"The number of valid items in {self.split} phase", "blue")
                    + f": {len(np.unique(item_feat))}",
                ]
            )
        return "\n".join(info)

    @property
    def sparsity_interaction_data_split(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        user_num_split = len(self._dataset.inter_feat[self._dataset.uid_field].unique())
        item_num_split = len(self._dataset.inter_feat[self._dataset.iid_field].unique())
        return 1 - self._dataset.inter_num / user_num_split / item_num_split

    @property
    def sparsity_knowledge_data_split(self):
        """Get the sparsity of this knowledge dataset.

        Returns:
            float: Sparsity of this knowledge dataset.
        """
        heads = len(self._dataset.kg_feat[self._dataset.kg_feat.head_id].unique())
        tails = len(self._dataset.kg_feat[self._dataset.kg_feat.tail_id].unique())
        return 1 - len(self._dataset.kg_feat) / heads / tails

    def avg_actions(self, feat, group_field):
        """Get the average number of entity heads in kg split.

        Returns:
            numpy.float64: Average number of entity heads in knowledge graph.
        """
        from collections import Counter

        import pandas as pd

        if isinstance(feat, pd.DataFrame):
            return np.mean(feat.groupby(group_field).size())
        else:
            return np.mean(list(Counter(feat[group_field].numpy()).values()))


class KnowledgePathEvalDataLoader(FullSortRecEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle)

        user_df = self.user_df[self.uid_field]
        ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
        inference_path_dataset = [
            dataset.path_token_separator.join(
                [
                    dataset.tokenizer.bos_token,
                    PathLanguageModelingTokenType.USER.token + str(uid.item()),
                    PathLanguageModelingTokenType.RELATION.token + str(ui_relation),
                ]
            )
            for uid in user_df
        ]
        inference_tokenized_dataset = dataset.tokenizer(
            inference_path_dataset, return_tensors="pt", add_special_tokens=False
        )
        self.inference_tokenized_dataset = Interaction(inference_tokenized_dataset.data)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        _, history_index, positive_u, positive_i = super().collate_fn(index)
        return self.inference_tokenized_dataset[index], history_index, positive_u, positive_i
