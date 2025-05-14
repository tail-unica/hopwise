# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""hopwise.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import joblib
import numpy as np
import torch
from tqdm import tqdm

from hopwise.data.dataset import KGSeqDataset, KnowledgePathDataset, SequentialDataset
from hopwise.data.interaction import Interaction
from hopwise.sampler import SeqSampler
from hopwise.utils import FeatureType, set_color


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~hopwise.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in
    2021/2/25, and he updated the codes in 2021/3/19. In 2021/7/9,
    Yupeng refactored SequentialDataset & SequentialDataLoader, then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in hopwise.
        seq_sample (hopwise.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field])

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = self.neg_item_list[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class KGGLMDataset(KnowledgePathDataset):
    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.train_stage = self.config["train_stage"]

        path_sample_args = self.config["path_sample_args"]
        self.pretrain_hop_length = path_sample_args["pretrain_hop_length"]
        self.pretrain_hop_length = tuple(map(int, self.pretrain_hop_length[1:-1].split(",")))
        self.pretrain_paths = path_sample_args["pretrain_paths"]

    def generate_user_path_dataset(self, used_ids):
        if self.train_stage == "lp_pretrain":
            self.generate_pretrain_dataset()
            if self.config["save_dataloaders"] or self.config["save_dataset"]:
                self.config["save_dataloaders"] = False
                self.config["save_dataset"] = False
                self.logger.warning(
                    "Pretraining dataset and dataloaders for KGGLM will not be saved. "
                    "Only with train-stage = finetune, the dataset and dataloaders will be saved."
                )
        else:
            super().generate_user_path_dataset(used_ids)

    def generate_pretrain_dataset(self):
        """Generate pretrain dataset for KGGLM model."""

        if self._path_dataset is None:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
            kg_rel_num = len(self.relations)
            graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num

            graph_min_iid = 1 + self.user_num
            min_hop, max_hop = self.pretrain_hop_length

            paths = set()
            iter_paths = tqdm(
                range(self.pretrain_paths),
                ncols=100,
                total=self.pretrain_paths,
                desc=set_color("KGGLM Pre-training Path Sampling", "red"),
            )

            def _generate_paths_random_walks():
                start_node = np.random.randint(graph_min_iid, len(graph.vs))
                path_hop_length = np.random.randint(min_hop, max_hop + 1)

                while True:
                    generated_path = graph.random_walk(start_node, path_hop_length, weights="weight")
                    generated_path = tuple(generated_path)
                    if generated_path not in paths:
                        break

                paths.add(generated_path)

            if not self.parallel_max_workers:
                for _ in iter_paths:
                    _generate_paths_random_walks()
            else:
                joblib.Parallel(n_jobs=self.parallel_max_workers, prefer="threads")(
                    joblib.delayed(_generate_paths_random_walks)() for _ in iter_paths
                )

            paths_with_relations = self._add_paths_relations(graph, paths)

            path_string = ""
            for path in paths_with_relations:
                path_string += self._format_path(path) + "\n"

            self._path_dataset = path_string
