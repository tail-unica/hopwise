# @Time   : 2020/9/3
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/16, 2020/9/15, 2020/10/25, 2022/7/10
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen, Lanling Xu
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn, xulanling_sherry@163.com

# UPDATE:
# @Time   : 2025
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.data.kg_dataset
##########################
"""

import copy
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from hopwise.data.dataset import Dataset
from hopwise.data.interaction import Interaction
from hopwise.utils import FeatureSource, FeatureType, KnowledgeEvaluationType, set_color
from hopwise.utils.url import decide_download, download_url, extract_zip


class KnowledgeBasedDataset(Dataset):
    """:class:`KnowledgeBasedDataset` is based on :class:`~hopwise.data.dataset.dataset.Dataset`,
    and load ``.kg`` and ``.link`` additionally.

    Entities are remapped together with ``item_id`` specially.
    All entities are remapped into three consecutive ID sections.

    - virtual entities that only exist in interaction data.
    - entities that exist both in interaction data and kg triplets.
    - entities only exist in kg triplets.

    It also provides several interfaces to transfer ``.kg`` features into coo sparse matrix,
    csr sparse matrix, :class:`DGL.Graph` or :class:`PyG.Data`.

    Attributes:
        head_entity_field (str): The same as ``config['HEAD_ENTITY_ID_FIELD']``.

        tail_entity_field (str): The same as ``config['TAIL_ENTITY_ID_FIELD']``.

        relation_field (str): The same as ``config['RELATION_ID_FIELD']``.

        entity_field (str): The same as ``config['ENTITY_ID_FIELD']``.

        kg_feat (pandas.DataFrame): Internal data structure stores the kg triplets.
            It's loaded from file ``.kg``.

        item2entity (dict): Dict maps ``item_id`` to ``entity``,
            which is loaded from  file ``.link``.

        entity2item (dict): Dict maps ``entity`` to ``item_id``,
            which is loaded from  file ``.link``.

    Note:
        :attr:`entity_field` doesn't exist exactly. It's only a symbol,
        representing entity features.

        :attr:`ui_relation` is a special relation token, which is used to represent
        the interaction relation between users and items.
    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.head_entity_field = self.config["HEAD_ENTITY_ID_FIELD"]
        self.tail_entity_field = self.config["TAIL_ENTITY_ID_FIELD"]
        self.relation_field = self.config["RELATION_ID_FIELD"]
        self.entity_field = self.config["ENTITY_ID_FIELD"]
        self.kg_reverse_r = self.config["kg_reverse_r"]
        self.ui_relation = self.config["ui_relation"]
        self.entity_kg_num_interval = self.config["entity_kg_num_interval"]
        self.relation_kg_num_interval = self.config["relation_kg_num_interval"]
        self._check_field("head_entity_field", "tail_entity_field", "relation_field", "entity_field")
        self.set_field_property(self.entity_field, FeatureType.TOKEN, FeatureSource.KG, 1)

        self.logger.debug(set_color("relation_field", "blue") + f": {self.relation_field}")
        self.logger.debug(set_color("entity_field", "blue") + f": {self.entity_field}")

    def _data_filtering(self):
        super()._data_filtering()
        self._filter_kg_by_triple_num()
        self._filter_link()

    def _filter_kg_by_triple_num(self):
        """Filter by number of triples.

        The interval of the number of triples can be set, and only entities/relations
        whose number of triples is in the specified interval can be retained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Lower bound of the interval is also called k-core filtering, which means this method
            will filter loops until all the entities and relations has at least k triples.
        """
        entity_kg_num_interval = self._parse_intervals_str(self.config["entity_kg_num_interval"])
        relation_kg_num_interval = self._parse_intervals_str(self.config["relation_kg_num_interval"])

        if entity_kg_num_interval is None and relation_kg_num_interval is None:
            return

        entity_kg_num = Counter()
        if entity_kg_num_interval:
            head_entity_kg_num = Counter(self.kg_feat[self.head_entity_field].values)
            tail_entity_kg_num = Counter(self.kg_feat[self.tail_entity_field].values)

            self.head_entity_kg_num = head_entity_kg_num
            entity_kg_num = head_entity_kg_num + tail_entity_kg_num
        self.entity_kg_num = entity_kg_num
        relation_kg_num = Counter(self.kg_feat[self.relation_field].values) if relation_kg_num_interval else Counter()

        while True:
            ban_head_entities = self._get_illegal_ids_by_inter_num(
                field=self.head_entity_field,
                feat=None,
                inter_num=entity_kg_num,
                inter_interval=entity_kg_num_interval,
            )
            ban_tail_entities = self._get_illegal_ids_by_inter_num(
                field=self.tail_entity_field,
                feat=None,
                inter_num=entity_kg_num,
                inter_interval=entity_kg_num_interval,
            )
            ban_entities = ban_head_entities | ban_tail_entities
            ban_relations = self._get_illegal_ids_by_inter_num(
                field=self.relation_field,
                feat=None,
                inter_num=relation_kg_num,
                inter_interval=relation_kg_num_interval,
            )
            if len(ban_entities) == 0 and len(ban_relations) == 0:
                break

            dropped_kg = pd.Series(False, index=self.kg_feat.index)
            head_entity_kg = self.kg_feat[self.head_entity_field]
            tail_entity_kg = self.kg_feat[self.tail_entity_field]
            relation_kg = self.kg_feat[self.relation_field]
            dropped_kg |= head_entity_kg.isin(ban_entities)
            dropped_kg |= tail_entity_kg.isin(ban_entities)
            dropped_kg |= relation_kg.isin(ban_relations)

            entity_kg_num -= Counter(head_entity_kg[dropped_kg].values)
            entity_kg_num -= Counter(tail_entity_kg[dropped_kg].values)
            relation_kg_num -= Counter(relation_kg[dropped_kg].values)

            dropped_index = self.kg_feat.index[dropped_kg]
            self.logger.debug(f"[{len(dropped_index)}] dropped triples.")
            self.kg_feat.drop(dropped_index, inplace=True)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~hopwise.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError("The ordering_method [{ordering_args}] has not been implemented.")

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        eval_lp_args = self.config["eval_lp_args"]

        if eval_lp_args is not None and eval_lp_args["knowledge_split"] is not None:
            knowledge_split_args = eval_lp_args["knowledge_split"]
            print("Splitting the knowledge graph")
            if not isinstance(knowledge_split_args, dict):
                raise ValueError(f"The knowledge_split_args [{knowledge_split_args}] should be a dict.")
            else:
                knowledge_split_mode = list(knowledge_split_args.keys())[0]
                assert len(knowledge_split_args.keys()) == 1
                knowledge_group_by = eval_lp_args["knowledge_group_by"]
        else:
            knowledge_split_mode = None
            knowledge_group_by = None

        # split_args is for interaction data
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]

        assert len(split_args.keys()) == 1

        group_by = self.config["eval_args"]["group_by"]

        datasets = dict()
        if knowledge_split_mode == "RS":
            # Manage knowledge graph split
            if not isinstance(knowledge_split_args["RS"], list):
                raise ValueError(
                    f'The value of "RS" in knowledge_split_args [{knowledge_split_args}] should be a list.'
                )

            if knowledge_group_by is not None:
                if knowledge_group_by.lower() == "head":
                    knowledge_group_by = self.head_entity_field
                elif knowledge_group_by.lower() == "tail":
                    knowledge_group_by = self.tail_entity_field
                elif knowledge_group_by.lower() == "relation":
                    knowledge_group_by = self.relation_field
                else:
                    raise NotImplementedError(
                        f"The knowledge grouping method [{knowledge_group_by}] has not been implemented."
                    )

            datasets[KnowledgeEvaluationType.LP] = self.split_by_ratio(
                knowledge_split_args["RS"],
                data={"data": self.kg_feat, "name": KnowledgeEvaluationType.LP},
                group_by=knowledge_group_by,
            )

        if split_mode == "RS":
            # Manage interaction split
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" in split_args [{split_args}] should be a list.')
            if group_by is None:
                datasets[KnowledgeEvaluationType.REC] = self.split_by_ratio(
                    split_args["RS"],
                    data={"data": self.inter_feat, "name": KnowledgeEvaluationType.REC},
                    group_by=None,
                )
            elif group_by.lower() == "user":
                datasets[KnowledgeEvaluationType.REC] = self.split_by_ratio(
                    split_args["RS"],
                    data={"data": self.inter_feat, "name": KnowledgeEvaluationType.REC},
                    group_by=self.uid_field,
                )
            else:
                raise NotImplementedError(f"The grouping method [{group_by}] has not been implemented.")
        elif split_mode == "LS":
            datasets[KnowledgeEvaluationType.REC] = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        else:
            raise NotImplementedError(f"The splitting_method [{split_mode}] has not been implemented.")
        return datasets[KnowledgeEvaluationType.REC] if KnowledgeEvaluationType.LP not in datasets else datasets

    def copy(self, new_inter_feat, data_type=KnowledgeEvaluationType.REC):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        if data_type == KnowledgeEvaluationType.REC:
            nxt.inter_feat = new_inter_feat
        else:
            nxt.kg_feat = new_inter_feat
        return nxt

    def split_by_ratio(self, ratios, data, group_by=None):
        """Split interaction records by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been split.

        Note:
            Other than the first one, each part is rounded down.
        """

        self.logger.debug(f"split {data['name']} by ratios [{ratios}], group_by=[{group_by}]")
        data_type = data["name"]
        data = data["data"]

        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]
        if group_by is None:
            split_ids = self._calcu_split_ids(tot=len(data), ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [len(data)])]

        else:
            grouped_data_feat_index = self._grouped_index(data[group_by].numpy())
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_data_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

        self._drop_unused_col()
        next_df = [data[index] for index in next_index]
        next_ds = [self.copy(split, data_type) for split in next_df]

        if data_type == KnowledgeEvaluationType.LP:
            # self.kg_feat now have only train data, to prevent data leakage
            self.kg_feat = next_df[0]
        return next_ds

    @property
    def tail_num(self):
        """Get the number of different tokens of ``self.tail_entity_field``.

        Returns:
            int: Number of different tokens of ``self.tail_entity_field``.
        """
        self._check_field("tail_entity_field")
        return self.num(self.tail_entity_field)

    def get_tail_feature(self):
        """Returns:
        Interaction: tails features
        """

        if self.tail_feat is None:
            self._check_field("tail_entity_field")
            return Interaction({self.tail_entity_field: torch.arange(self.tail_num)})
        else:
            return self.tail_feat

    def _filter_link(self):
        """Filter rows of :attr:`item2entity` and :attr:`entity2item`,
        whose ``entity_id`` doesn't occur in kg triplets and
        ``item_id`` doesn't occur in interaction records.
        """
        item_tokens = self._get_rec_item_token()
        ent_tokens = self._get_entity_token()
        illegal_item = set()
        illegal_ent = set()
        for item in self.item2entity:
            ent = self.item2entity[item]
            if item not in item_tokens or ent not in ent_tokens:
                illegal_item.add(item)
                illegal_ent.add(ent)
        for item in illegal_item:
            del self.item2entity[item]
        for ent in illegal_ent:
            del self.entity2item[ent]
        remained_inter = pd.Series(True, index=self.inter_feat.index)
        remained_inter &= self.inter_feat[self.iid_field].isin(self.item2entity.keys())
        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _download(self):
        super()._download()

        url = self._get_download_url("kg_url", allow_none=True)
        if url is None:
            return
        self.logger.info(f"Prepare to download linked knowledge graph from [{url}].")

        if decide_download(url):
            # No need to create dir, as `super()._download()` has created one.
            path = download_url(url, self.dataset_path)
            extract_zip(path, self.dataset_path)
            os.unlink(path)
            self.logger.info(
                f"\nLinked KG for [{self.dataset_name}] requires additional conversion "
                f"to atomic files (.kg and .link).\n"
                f"Please refer to https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools#knowledge-aware-datasets "  # noqa: E501
                f"for detailed instructions.\n"
                f"You can run hopwise after the conversion, see you soon."
            )
            sys.exit(0)
        else:
            self.logger.info("Stop download.")
            sys.exit(-1)

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.kg_feat = self._load_kg(self.dataset_name, self.dataset_path)
        self.tail_feat = None
        self.item2entity, self.entity2item = self._load_link(self.dataset_name, self.dataset_path)

    @property
    def kg_num(self):
        """Get the number of interaction records.

        Returns:
            int: Number of interaction records.
        """
        return len(self.kg_feat)

    @property
    def sparsity_kg(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.kg_num / (self.entity_num**2)

    @property
    def sparsity_kg_rel(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.kg_num / (self.entity_num**2 * self.relation_num)

    @property
    def avg_degree_kg_item(self):
        """Get the average degree of items in the knowledge graph.

        Returns:
            float: Average number of KG triples each item is involved in.
        """  # assumes a DataFrame or dict with head, relation, tail
        if isinstance(self.kg_feat, pd.DataFrame):
            head_counts = self.kg_feat[self.head_entity_field].value_counts()
            tail_counts = self.kg_feat[self.tail_entity_field].value_counts()
            total_counts = head_counts.add(tail_counts, fill_value=0)
            item_degrees = total_counts[total_counts.index.astype(str).isin(self.item2entity.keys())]
            return item_degrees.mean() if not item_degrees.empty else 0.0
        else:
            # fallback if not using pandas
            head = self.kg_feat[self.head_entity_field].numpy()
            tail = self.kg_feat[self.tail_entity_field].numpy()
            counter = Counter(head) + Counter(tail)
            item_degrees = [counter[pid] for pid in self.item2entity.keys()]
            return np.mean(item_degrees) if item_degrees else 0.0

    @property
    def avg_degree_kg(self):
        """Get the average degree of all entities in the knowledge graph.

        Returns:
            float: Average number of triples each entity is involved in.
        """
        return 2 * self.kg_num / self.entity_num

    def __str__(self):
        info = [
            super().__str__(),
            set_color("The number of entities","green") + f": {self.entity_num}",
            set_color("The number of relations","green")+ f": {self.relation_num}",
            set_color("The number of triples","green")+ f": {self.kg_num}",
            set_color("The number of items that have been linked to KG", "green") + f": {len(self.item2entity)}",
            set_color("The number of items that have not been linked to KG",
                      "green") + f": {self.item_num - len(self.item2entity)}",
            set_color("The sparsity of the KG","green") + f": {self.sparsity_kg_rel}",
            set_color("The sparsity of the KG (relation-aware)","green") + f": {self.sparsity_kg}",
            set_color("The average degree of entities in the KG:","green") + f": {self.avg_degree_kg}",
            set_color("The average degree of items in the KG:","green") + f": {self.avg_degree_kg_item}",
        ]  # yapf: disable
        return "\n".join(info)

    def _build_feat_name_list(self):
        feat_name_list = super()._build_feat_name_list()
        if self.kg_feat is not None:
            feat_name_list.append("kg_feat")
        return feat_name_list

    def _load_kg(self, token, dataset_path):
        self.logger.debug(set_color(f"Loading kg from [{dataset_path}].", "green"))
        kg_path = os.path.join(dataset_path, f"{token}.kg")
        if not os.path.isfile(kg_path):
            raise ValueError(f"[{token}.kg] not found in [{dataset_path}].")
        df = self._load_feat(kg_path, FeatureSource.KG)
        self._check_kg(df)
        return df

    def _check_kg(self, kg):
        kg_warn_message = "kg data requires field [{}]"
        assert self.head_entity_field in kg, kg_warn_message.format(self.head_entity_field)
        assert self.tail_entity_field in kg, kg_warn_message.format(self.tail_entity_field)
        assert self.relation_field in kg, kg_warn_message.format(self.relation_field)

    def _load_link(self, token, dataset_path):
        self.logger.debug(set_color(f"Loading link from [{dataset_path}].", "green"))
        link_path = os.path.join(dataset_path, f"{token}.link")
        if not os.path.isfile(link_path):
            raise ValueError(f"[{token}.link] not found in [{dataset_path}].")
        df = self._load_feat(link_path, "link")
        self._check_link(df)

        item2entity, entity2item = {}, {}
        for item_id, entity_id in zip(df[self.iid_field].values, df[self.entity_field].values):
            item2entity[item_id] = entity_id
            entity2item[entity_id] = item_id
        return item2entity, entity2item

    def _check_link(self, link):
        link_warn_message = "link data requires field [{}]"
        assert self.entity_field in link, link_warn_message.format(self.entity_field)
        assert self.iid_field in link, link_warn_message.format(self.iid_field)

    def _init_alias(self):
        """Add :attr:`alias_of_entity_id`, :attr:`alias_of_relation_id` and update :attr:`_rest_fields`."""
        self._set_alias("entity_id", [self.head_entity_field, self.tail_entity_field])
        self._set_alias("relation_id", [self.relation_field])

        super()._init_alias()

        self._rest_fields = np.setdiff1d(self._rest_fields, [self.entity_field], assume_unique=True)

    def _get_rec_item_token(self):
        """Get set of entity tokens from fields in ``rec`` level."""
        remap_list = self._get_remap_list(self.alias["item_id"])
        tokens, _ = self._concat_remaped_tokens(remap_list)
        return set(tokens)

    def _get_entity_token(self):
        """Get set of entity tokens from fields in ``ent`` level."""
        remap_list = self._get_remap_list(self.alias["entity_id"])
        tokens, _ = self._concat_remaped_tokens(remap_list)
        return set(tokens)

    def _reset_ent_remapID(self, field, idmap, id2token, token2id):
        self.field2id_token[field] = id2token
        self.field2token_id[field] = token2id
        for feat in self.field2feats(field):
            ftype = self.field2type[field]
            if ftype == FeatureType.TOKEN:
                old_idx = feat[field].values
            else:
                old_idx = feat[field].agg(np.concatenate)

            new_idx = idmap[old_idx]

            if ftype == FeatureType.TOKEN:
                feat[field] = new_idx
            else:
                split_point = np.cumsum(feat[field].transform(len))[:-1]
                feat[field] = np.split(new_idx, split_point)

    def _merge_item_and_entity(self):
        """Merge item-id and entity-id into the same id-space."""
        item_token = self.field2id_token[self.iid_field]
        entity_token = self.field2id_token[self.head_entity_field]
        item_num = len(item_token)
        link_num = len(self.item2entity)
        entity_num = len(entity_token)

        # reset item id
        item_priority = np.array([token in self.item2entity for token in item_token])
        item_order = np.argsort(item_priority, kind="stable")
        item_id_map = np.zeros_like(item_order)
        item_id_map[item_order] = np.arange(item_num)
        new_item_id2token = item_token[item_order]
        new_item_token2id = {t: i for i, t in enumerate(new_item_id2token)}
        for field in self.alias["item_id"]:
            self._reset_ent_remapID(field, item_id_map, new_item_id2token, new_item_token2id)

        # reset entity id
        entity_priority = np.array([token != "[PAD]" and token not in self.entity2item for token in entity_token])
        entity_order = np.argsort(entity_priority, kind="stable")
        entity_id_map = np.zeros_like(entity_order)
        for i in entity_order[1 : link_num + 1]:
            entity_id_map[i] = new_item_token2id[self.entity2item[entity_token[i]]]
        entity_id_map[entity_order[link_num + 1 :]] = np.arange(item_num, item_num + entity_num - link_num - 1)
        new_entity_id2token = np.concatenate([new_item_id2token, entity_token[entity_order[link_num + 1 :]]])
        for i in range(item_num - link_num, item_num):
            new_entity_id2token[i] = self.item2entity[new_entity_id2token[i]]
        new_entity_token2id = {t: i for i, t in enumerate(new_entity_id2token)}
        for field in self.alias["entity_id"]:
            self._reset_ent_remapID(field, entity_id_map, new_entity_id2token, new_entity_token2id)
        self.field2id_token[self.entity_field] = new_entity_id2token
        self.field2token_id[self.entity_field] = new_entity_token2id

    def _add_auxiliary_relation(self):
        """Add auxiliary relations in ``self.relation_field``."""
        if self.kg_reverse_r:
            # '0' is used for padding, so the number needs to be reduced by one
            original_rel_num = len(self.field2id_token[self.relation_field]) - 1
            original_hids = self.kg_feat[self.head_entity_field]
            original_tids = self.kg_feat[self.tail_entity_field]
            original_rels = self.kg_feat[self.relation_field]

            # Internal id gap of a relation and its reverse edge is original relation num
            reverse_rels = original_rels + original_rel_num

            # Add mapping for internal and external ID of relations
            for i in range(1, original_rel_num + 1):
                original_token = self.field2id_token[self.relation_field][i]

                # ui_relation may already exist in the relation field when using pre-trained embeddings
                if original_token == self.ui_relation:
                    continue

                reverse_token = original_token + "_r"
                self.field2token_id[self.relation_field][reverse_token] = i + original_rel_num
                self.field2id_token[self.relation_field] = np.append(
                    self.field2id_token[self.relation_field], reverse_token
                )

            # Update knowledge graph triples with reverse relations
            reverse_kg_data = {
                self.head_entity_field: original_tids,
                self.relation_field: reverse_rels,
                self.tail_entity_field: original_hids,
            }
            reverse_kg_feat = pd.DataFrame(reverse_kg_data)
            self.kg_feat = pd.concat([self.kg_feat, reverse_kg_feat])

        # Add UI-relation pairs in the relation field
        if self.ui_relation not in self.field2token_id[self.relation_field]:
            kg_rel_num = len(self.field2id_token[self.relation_field])
            self.field2token_id[self.relation_field][self.ui_relation] = kg_rel_num
            self.field2id_token[self.relation_field] = np.append(
                self.field2id_token[self.relation_field], self.ui_relation
            )

    def _remap_ID_all(self):
        super()._remap_ID_all()
        self._merge_item_and_entity()
        self._add_auxiliary_relation()

    @property
    def relation_num(self):
        """Get the number of different tokens of ``self.relation_field``.

        Returns:
            int: Number of different tokens of ``self.relation_field``.
        """
        return self.num(self.relation_field)

    @property
    def entity_num(self):
        """Get the number of different tokens of entities, including virtual entities.

        Returns:
            int: Number of different tokens of entities, including virtual entities.
        """
        return self.num(self.entity_field)

    @property
    def head_entities(self):
        """Returns:
        numpy.ndarray: List of head entities of kg triplets.
        """
        return self.kg_feat[self.head_entity_field].numpy()

    @property
    def tail_entities(self):
        """Returns:
        numpy.ndarray: List of tail entities of kg triplets.
        """
        return self.kg_feat[self.tail_entity_field].numpy()

    @property
    def relations(self):
        """Returns:
        numpy.ndarray: List of relations of kg triplets.
        """
        return self.kg_feat[self.relation_field].numpy()

    def norm_ckg_adjacency_matrix(self, form="torch_sparse"):
        """Get the collaborative normalized adjacency matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Args:
            form (str, optional): Format of the normalized adjacency matrix. Defaults to ``torch_sparse``.

        Returns:
            torch.sparse.FloatTensor: Normalized adjacency matrix.

        Raises:
            NotImplementedError: If the format of the normalized adjacency matrix is not implemented.
        """
        if form == "torch_sparse":
            return self._create_norm_ckg_adjacency_matrix()
        else:
            raise NotImplementedError(f"Normalized adjacency matrix format [{form}] has not been implemented.")

    def _create_norm_ckg_adjacency_matrix(self, size=None, symmetric=True):
        """Get the normalized interaction matrix of users and entities (items) and
        the normalized adjacency matrix of the collaborative knowledge graph.

        Uses :func:`~hopwise.data.dataset.dataset.Dataset._create_norm_adjacency_matrix`
        to get the normalized adjacency matrix of the collaborative knowledge graph
        and then extract the normalized interaction matrix of users and entities (items).

        Returns:
            tuple: tuple of:
            - normalized interaction matrix of users and entities (items)
            - normalized adjacency matrix of the collaborative knowledge graph.

        """
        if size is None:
            size = self.user_num + self.entity_num

        norm_graph = self._create_norm_adjacency_matrix(size=size, symmetric=symmetric)
        if not norm_graph.is_coalesced():
            norm_graph = norm_graph.coalesce()

        row, col = norm_graph.indices().cpu().numpy()
        values = norm_graph.values().cpu().numpy()
        mat = coo_matrix((values, (row, col)), shape=tuple(norm_graph.shape))
        norm_matrix = mat.tocsr()[: self.user_num, self.user_num :].tocoo()

        indices = torch.LongTensor(np.array([norm_matrix.row, norm_matrix.col]))
        data = torch.FloatTensor(norm_matrix.data)
        norm_matrix = torch.sparse.FloatTensor(indices, data, norm_matrix.shape)

        return norm_matrix, norm_graph

    @property
    def entities(self):
        """Returns:
        numpy.ndarray: List of entity id, including virtual entities.
        """
        return np.arange(self.entity_num)

    def kg_graph(self, form="coo", value_field=None):
        """Get graph or sparse matrix that describe relations between entities.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = self.kg_feat[value_field][src, tgt]``.

        Currently, we support graph in `DGL`_ and `PyG`_,
        and two type of sparse matrices, ``coo`` and ``csr``.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``coo``.
            value_field (str, optional): edge attributes of graph, or data of sparse matrix,
                Defaults to ``None``.

        Returns:
            Graph / Sparse matrix of kg triplets.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        args = [
            self.kg_feat,
            self.head_entity_field,
            self.tail_entity_field,
            form,
            value_field,
        ]
        if form in ["coo", "csr"]:
            return self._create_sparse_matrix(*args)
        elif form in ["dgl", "pyg"]:
            return self._create_graph(*args)
        else:
            raise NotImplementedError("kg graph format [{}] has not been implemented.")

    def _create_ckg_source_target(self, form="numpy"):
        """Create base collaborative knowledge graph.

        Args:
            form (str, optional): The format of the returned graph source and target.
            Defaults to ``numpy``.
        """
        user_num = self.user_num

        if form == "numpy":
            hids = self.head_entities + user_num
            tids = self.tail_entities + user_num

            uids = self.inter_feat[self.uid_field].numpy()
            iids = self.inter_feat[self.iid_field].numpy() + user_num
            src = np.concatenate([uids, iids, hids])
            tgt = np.concatenate([iids, uids, tids])
        elif form == "torch":
            kg_tensor = self.kg_feat
            inter_tensor = self.inter_feat

            hids = kg_tensor[self.head_entity_field] + user_num
            tids = kg_tensor[self.tail_entity_field] + user_num

            uids = inter_tensor[self.uid_field]
            iids = inter_tensor[self.iid_field] + user_num

            src = torch.cat([uids, iids, hids])
            tgt = torch.cat([iids, uids, tids])
        else:
            raise NotImplementedError(f"form [{form}] has not been implemented.")

        return src, tgt

    def _create_ckg_sparse_matrix(self, form="coo", show_relation=False):
        src, tgt = self._create_ckg_source_target(form="numpy")

        ui_rel_num = self.inter_num
        ui_rel_id = self.relation_num - 1
        assert self.field2id_token[self.relation_field][ui_rel_id] == self.ui_relation

        if not show_relation:
            data = np.ones(len(src))
        else:
            kg_rel = self.kg_feat[self.relation_field].numpy()
            ui_rel = np.full(2 * ui_rel_num, ui_rel_id, dtype=kg_rel.dtype)
            data = np.concatenate([ui_rel, kg_rel])
        node_num = self.entity_num + self.user_num
        mat = coo_matrix((data, (src, tgt)), shape=(node_num, node_num))
        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()
        else:
            raise NotImplementedError(f"Sparse matrix format [{form}] has not been implemented.")

    def _create_ckg_graph(self, form="dgl", show_relation=False):
        src, tgt = self._create_ckg_source_target(form="torch")

        if show_relation:
            ui_rel_num = len(self.inter_feat)

            ui_rel_id = self.field2token_id[self.relation_field][self.ui_relation]

            kg_rel = self.kg_feat[self.relation_field]
            ui_rel = torch.full((2 * ui_rel_num,), ui_rel_id, dtype=kg_rel.dtype)
            edge = torch.cat([ui_rel, kg_rel])

        if form == "dgl":
            import dgl

            graph = dgl.graph((src, tgt))
            if show_relation:
                graph.edata[self.relation_field] = edge
            return graph
        elif form == "pyg":
            from torch_geometric.data import Data

            edge_attr = edge if show_relation else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError(f"Graph format [{form}] has not been implemented.")

    def _create_ckg_igraph(self, show_relation=False, directed=True):
        import igraph as ig

        vertex_type_attrs = np.concatenate(
            [
                [self.uid_field] * self.user_num,
                [self.iid_field] * self.item_num,
                [self.entity_field] * (self.entity_num - self.item_num),
            ],
            axis=0,
        )
        if show_relation:
            n_ui_relations = self.inter_num * 2 if directed else self.inter_num
            edge_type_attrs = np.concatenate(
                [[self.ui_relation] * n_ui_relations, self.field2id_token[self.relation_field][self.relations]], axis=0
            )
        else:
            edge_type_attrs = None

        if directed:
            src, tgt = self._create_ckg_source_target(form="numpy")
        else:
            user_num = self.user_num
            hids = self.head_entities + user_num
            tids = self.tail_entities + user_num

            uids = self.inter_feat[self.uid_field].numpy()
            iids = self.inter_feat[self.iid_field].numpy() + user_num

            src = np.concatenate([uids, hids])
            tgt = np.concatenate([iids, tids])

        tuple_graph = list(zip(src, tgt))
        ig_graph = ig.Graph(
            edges=tuple_graph,
            vertex_attrs={"type": vertex_type_attrs},
            edge_attrs={"type": edge_type_attrs} if show_relation else None,
            directed=directed,
        )

        return ig_graph

    def ckg_graph(self, form="coo", value_field=None):
        """Get graph or sparse matrix that describe relations of CKG,
        which combines interactions and kg triplets into the same graph.

        Item ids and entity ids are added by ``user_num`` temporally.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = self.kg_feat[self.relation_field][src, tgt]``
        or ``graph[src, tgt] = self.ui_relation``.

        Currently, we support graph in `DGL`_, `PyG`_, and `igraph`_,
        two type of sparse matrices, ``coo`` and ``csr``.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``coo``.
            value_field (str, optional): ``self.relation_field`` or ``None``,
                Defaults to ``None``.

        Returns:
            Graph / Sparse matrix of kg triplets.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric

        .. _igraph:
            https://python.igraph.org/en/stable/
        """
        if value_field is not None and value_field != self.relation_field:
            raise ValueError(f"Value_field [{value_field}] can only be [{self.relation_field}] in ckg_graph.")
        show_relation = value_field is not None

        if form in ["coo", "csr"]:
            return self._create_ckg_sparse_matrix(form, show_relation)
        elif form in ["dgl", "pyg"]:
            return self._create_ckg_graph(form, show_relation)
        elif form == "igraph":
            return self._create_ckg_igraph(show_relation)
        else:
            raise NotImplementedError("ckg graph format [{}] has not been implemented.")

    def _create_hetero_ckg_graph(self, form="dgl", directed=False):
        """DGL expects each node type to be in the range [0, num_nodes_dict[ntype])."""
        import dgl

        item_num = self.item_num
        inter_tensor = self.inter_feat
        kg_tensor = self.kg_feat

        uids = inter_tensor[self.uid_field]
        iids = inter_tensor[self.iid_field]

        graph_data = {(self.uid_field, self.ui_relation, self.iid_field): (uids, iids)}
        if not directed:
            graph_data[(self.iid_field, self.ui_relation, self.uid_field)] = (iids, uids)

        hids = kg_tensor[self.head_entity_field]
        tids = kg_tensor[self.tail_entity_field]
        kg_rel = kg_tensor[self.relation_field]
        entity_token = self.field2id_token[self.entity_field]
        entities_not_items_mask = np.array(
            [token != "[PAD]" and token not in self.entity2item for token in entity_token]
        )
        for rel, rel_id in self.field2token_id[self.relation_field].items():
            if rel in ["[PAD]", self.ui_relation]:
                continue

            rel_mask = kg_rel == rel_id
            rel_hids = hids[rel_mask]
            rel_tids = tids[rel_mask]

            rel_hids_ents = np.take(entities_not_items_mask, rel_hids)
            rel_tids_ents = np.take(entities_not_items_mask, rel_tids)

            # Entity-entity links
            entity_entity_links = np.logical_and(rel_hids_ents, rel_tids_ents)
            if entity_entity_links.any():
                ee_hids = rel_hids[entity_entity_links] - item_num
                ee_tids = rel_tids[entity_entity_links] - item_num
                graph_data[(self.entity_field, rel, self.entity_field)] = (ee_hids, ee_tids)

            # Entity-item links
            entity_item_links = np.logical_and(rel_hids_ents, ~rel_tids_ents)
            if entity_item_links.any():
                ei_hids = rel_hids[entity_item_links] - item_num
                ei_tids = rel_tids[entity_item_links]
                graph_data[(self.entity_field, rel, self.iid_field)] = (ei_hids, ei_tids)

            # Item-entity links
            item_entity_links = np.logical_and(~rel_hids_ents, rel_tids_ents)
            if item_entity_links.any():
                ie_hids = rel_hids[item_entity_links]
                ie_tids = rel_tids[item_entity_links] - item_num
                graph_data[(self.iid_field, rel, self.entity_field)] = (ie_hids, ie_tids)

            # Item-item links
            item_item_links = np.logical_and(~rel_hids_ents, ~rel_tids_ents)
            if item_item_links.any():
                ii_hids = rel_hids[item_item_links]
                ii_tids = rel_tids[item_item_links]
                graph_data[(self.iid_field, rel, self.iid_field)] = (ii_hids, ii_tids)

        num_nodes_dict = {
            self.uid_field: self.user_num,
            self.iid_field: item_num,
            self.entity_field: self.entity_num - item_num,
        }
        graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

        return graph

    def ckg_hetero_graph(self, form="dgl", directed=False):
        """Get heterogeneous graph that describes relations of CKG,
        which does not only combine interactions and kg triplets into the same graph,
        but it also enable metapath-based random walks.

        Item ids and entity ids are added by ``user_num`` temporally.

        Currently, we support graph in `DGL`_.

        Args:
            form (str, optional): Format of sparse matrix, or library of graph data structure.
                Defaults to ``dgl``.
            directed (bool, optional): Whether the graph is directed or not.
                Defaults to ``False``.

        Returns:
            Heterogeneous graph.

        .. _DGL:
            https://www.dgl.ai/
        """

        if form in ["dgl"]:
            return self._create_hetero_ckg_graph(form, directed=directed)
        else:
            raise NotImplementedError("ckg hetero graph format [{}] has not been implemented.")

    def ckg_dict_graph(self, ui_bidirectional=True):
        """Get a dictionary representation of the collaborative knowledge graph.

        Returns:
            dict: Dictionary representation of the collaborative knowledge graph.
        """
        uids = self.inter_feat[self.uid_field].numpy()
        iids = self.inter_feat[self.iid_field].numpy()

        src = np.concatenate([uids, self.head_entities])
        tgt = np.concatenate([iids, self.tail_entities])

        ui_relation_id = self.field2token_id[self.relation_field][self.ui_relation]
        rels = np.concatenate([np.full(self.inter_num, ui_relation_id), self.relations])

        graph_dict = {"user": {}, "entity": {}}
        for idx, (src_id, rel_id, tgt_id) in enumerate(zip(src, rels, tgt)):
            if rel_id == ui_relation_id:
                src_type = "user"
                end_type = "entity"

                if src_id not in graph_dict[src_type]:
                    graph_dict[src_type][src_id] = dict()
                if rel_id not in graph_dict[src_type][src_id]:
                    graph_dict[src_type][src_id][rel_id] = list()

                # UI interaction case
                graph_dict[src_type][src_id][rel_id].append(tgt_id)
                if ui_bidirectional:
                    if tgt_id not in graph_dict[end_type]:
                        graph_dict[end_type][tgt_id] = dict()
                    if rel_id not in graph_dict[end_type][tgt_id]:
                        graph_dict[end_type][tgt_id][rel_id] = list()

                    graph_dict[end_type][tgt_id][rel_id].append(src_id)

            else:
                if src_id not in graph_dict["entity"]:
                    graph_dict["entity"][src_id] = dict()
                if rel_id not in graph_dict["entity"][src_id]:
                    graph_dict["entity"][src_id][rel_id] = list()

                if tgt_id not in graph_dict["entity"]:
                    graph_dict["entity"][tgt_id] = dict()
                if rel_id not in graph_dict["entity"][tgt_id]:
                    graph_dict["entity"][tgt_id][rel_id] = list()

                # KG case
                graph_dict["entity"][src_id][rel_id].append(tgt_id)
                graph_dict["entity"][tgt_id][rel_id].append(src_id)

        return graph_dict
