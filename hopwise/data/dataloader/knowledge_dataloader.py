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
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict

from hopwise.data.dataloader.abstract_dataloader import AbstractDataLoader
from hopwise.data.dataloader.general_dataloader import FullSortRecEvalDataLoader, TrainDataLoader
from hopwise.data.interaction import Interaction
from hopwise.utils import KGDataLoaderState, PathLanguageModelingTokenType


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
        self.logger = getLogger()
        # using sampler
        self.general_dataloader = TrainDataLoader(config, dataset, sampler, shuffle=shuffle)

        # using kg_sampler
        self.kg_dataloader = KGDataLoader(config, dataset, kg_sampler, shuffle=True)

        self.shuffle = False
        self.state = None
        self.dataset = self._dataset = dataset
        self.kg_iter, self.gen_iter = None, None

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


class KnowledgePathDataLoader(KnowledgeBasedDataLoader):
    """:class:`KnowledgePathDataLoader` is a dataloader for path-language-modeling on knowledge graphs.

    It mainly serves as a wrapper to :class:`~hopwise.data.dataset.kg_path_dataset.KnowledgePathDataset`,
    so as to be aware of generating paths only for the training phase.
    class:'KnowledgeBasedDataLoader' is subclassed to preserve the API of the original dataloader.

    Path dataset is tokenized at a later stage, so that the post_processor can be manipulated
    to add special tokens for path language modeling, e.g. token type ids.
    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        kg_sampler (KGSampler): The knowledge graph sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    Attributes:
        tokenized_dataset (DatasetDict): The tokenized path dataset.
    """

    def __init__(self, config, dataset, sampler, kg_sampler, shuffle=False):
        super().__init__(config, dataset, sampler, kg_sampler, shuffle=shuffle)
        self._tokenized_dataset = None
        # needs to be pre-generated
        self._dataset.generate_user_path_dataset(sampler.used_ids)
        # path_hop_length = n_relations => (n_relations + user_starting_node) + n_relations + 2 (BOS, EOS)
        self.token_sequence_length = (1 + dataset.path_hop_length) + dataset.path_hop_length + 2

        dataset.used_ids = self.general_dataloader._sampler.used_ids

    @property
    def tokenized_dataset(self):
        if self._tokenized_dataset is None:
            self.tokenize_path_dataset(phase="train")
        return self._tokenized_dataset

    def tokenize_path_dataset(self, phase="train"):
        """Tokenize the path dataset.

        Args:
            phase (str, optional): The phase for which the path dataset is used. Defaults to "train".
        """

        if self._tokenized_dataset is None:

            def remove_incorrect_paths(tokenized_dataset):
                # remove paths that contain special tokens. The token in position 0 and -1 are [BOS] and [EOS]
                return not any(
                    path in self._dataset.tokenizer.all_special_ids
                    for path in tokenized_dataset["input_ids"][1:-1]  # noqa: E501
                )

            def tokenization(example):
                return self._dataset.tokenizer(
                    example["path"],
                    truncation=True,
                    padding=True,
                    max_length=self._dataset.context_length,
                    add_special_tokens=True,
                )

            hf_path_dataset = HuggingFaceDataset.from_dict({"path": self._dataset.path_dataset.split("\n")})
            tokenized_dataset = hf_path_dataset.map(tokenization, batched=True, remove_columns=["path"])
            tokenized_dataset = tokenized_dataset.filter(remove_incorrect_paths)
            tokenized_dataset = DatasetDict({phase: tokenized_dataset})
            self._tokenized_dataset = tokenized_dataset

    def get_tokenized_used_ids(self):
        """Convert the used ids to tokenized ids.

        Args:
            used_ids (dict): A dictionary where keys are user ids and values are lists of item ids.
            tokenizer: The tokenizer to convert ids to tokenized ids.
        Returns:
            dict: A dictionary where keys are tokenized user ids and values are lists of tokenized item ids.
        """
        user_token_type = PathLanguageModelingTokenType.USER.value
        item_token_type = PathLanguageModelingTokenType.ITEM.value

        used_ids = self.general_dataloader._sampler.used_ids
        tokenizer = self._dataset.tokenizer

        tokenized_used_ids = {}
        for uid in range(used_ids.shape[0]):
            uid_token = tokenizer.convert_tokens_to_ids(user_token_type + str(uid))
            tokenized_used_ids[uid_token] = set(
                [tokenizer.convert_tokens_to_ids(item_token_type + str(item)) for item in used_ids[uid]]
            )
        return tokenized_used_ids


class KnowledgePathEvalDataLoader(FullSortRecEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle)

        from datasets import Dataset

        user_df = self.user_df[self.uid_field]
        ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
        inference_path_dataset = {
            self.uid_field: [
                dataset.path_token_separator.join(
                    [
                        dataset.tokenizer.bos_token,
                        PathLanguageModelingTokenType.USER.value + str(uid.item()),
                        PathLanguageModelingTokenType.RELATION.value + str(ui_relation),
                    ]
                )
                for uid in user_df
            ]
        }
        self.inference_path_dataset = Dataset.from_dict(inference_path_dataset)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        _, history_index, positive_u, positive_i = super().collate_fn(index)
        return self.inference_path_dataset[index][self.uid_field], history_index, positive_u, positive_i
