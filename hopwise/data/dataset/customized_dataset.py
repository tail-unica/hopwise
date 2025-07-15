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

import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture as GMM

from hopwise.data.dataset import KGSeqDataset, KnowledgeBasedDataset, KnowledgePathDataset, SequentialDataset
from hopwise.data.interaction import Interaction
from hopwise.sampler import SeqSampler
from hopwise.utils import FeatureType, progress_bar, set_color


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

    def generate_user_path_dataset(self):
        if self.train_stage == "pretrain":
            self.generate_pretrain_dataset()
        else:
            super().generate_user_path_dataset()

    def generate_pretrain_dataset(self):
        """Generate pretrain dataset for KGGLM model."""

        if self._path_dataset is None:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
            kg_rel_num = len(self.relations)
            graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num

            graph_min_iid = 1 + self.user_num
            min_hop, max_hop = self.pretrain_hop_length

            paths = set()
            iter_paths = progress_bar(
                range(graph_min_iid + 1, len(graph.vs)),
                ncols=100,
                total=len(graph.vs) - graph_min_iid,
                desc=set_color("KGGLM Pre-training Path Sampling", "red", progress=True),
            )
            max_tries_per_entity = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]

            kwargs = dict(
                graph=graph,
                min_hop=min_hop,
                max_hop=max_hop,
                pretrain_paths=self.pretrain_paths,
                max_tries_per_entity=max_tries_per_entity,
                paths=paths,
            )

            if not self.parallel_max_workers:
                for entity in iter_paths:
                    _generate_paths_random_walks(entity)
            else:
                joblib.Parallel(n_jobs=self.parallel_max_workers, prefer="threads", return_as="generator")(
                    joblib.delayed(_generate_paths_random_walks)(entity, **kwargs) for entity in iter_paths
                )

            paths_with_relations = self._add_paths_relations(graph, paths)

            path_string = ""
            for path in paths_with_relations:
                path_string += self._format_path(path) + "\n"

            self._path_dataset = path_string


def _generate_paths_random_walks(start_node, **kwargs):
    graph = kwargs.get("graph")
    min_hop = kwargs.get("min_hop")
    max_hop = kwargs.get("max_hop")
    pretrain_paths = kwargs.get("pretrain_paths")
    max_tries_per_entity = kwargs.get("max_tries_per_entity")
    paths = kwargs.get("paths")

    for _ in range(pretrain_paths):
        path_hop_length = np.random.randint(min_hop, max_hop + 1)
        tries_per_entity = max_tries_per_entity

        while tries_per_entity > 0:
            generated_path = graph.random_walk(start_node, path_hop_length, weights="weight")
            generated_path = tuple(generated_path)
            if generated_path not in paths:
                break
            tries_per_entity -= 1

        paths.add(generated_path)


class TPRecTimestampDataset:
    """
    A class to create a clustered dataset based on interaction timestamp specifically for the TPRec model
    """

    Y = 2000
    seasons = [
        (0, (datetime.date(Y, 1, 1), datetime.date(Y, 3, 20))),  # 'winter'
        (1, (datetime.date(Y, 3, 21), datetime.date(Y, 6, 20))),  # 'spring'
        (2, (datetime.date(Y, 6, 21), datetime.date(Y, 9, 22))),  # 'summer'
        (3, (datetime.date(Y, 9, 23), datetime.date(Y, 12, 20))),  # 'autumn'
        (0, (datetime.date(Y, 12, 21), datetime.date(Y, 12, 31))),
    ]  # 'winter'

    def __init__(self, config, inter_feat, set="train", gmm=None):
        self.config = config
        self.inter_feat = inter_feat
        self.set = set
        data = {"users": inter_feat.user_id, "item": inter_feat.item_id, "timestamps": inter_feat.timestamp}
        self.data = pd.DataFrame(data)
        self.data.timestamps = self.data.timestamps.astype(int)
        self.data.timestamps = pd.to_datetime(self.data.timestamps, unit="s").dt.date

        self.user_item_timestamp = np.array(self.data.timestamps)
        time2num = self._timeanalysis()

        if config["cluster_feature"] == "all":
            fileTime = self._get_all_cluster_feature(time2num)
        elif config["cluster_feature"] == "w-stat":
            fileTime = self._get_w_stat_cluster_feature()
        elif config["cluster_feature"] == "w-stru":
            fileTime = self._get_w_stru_cluster_feature(time2num)
        else:
            raise ValueError(
                f"Unsupported cluster_feature: {config['cluster_feature']}. "
                "Available options are 'all', 'w-stat', 'w-stru'."
            )

        if self.set == "train":
            gmmModel, timeNum, labels, timeClassifyLabel = self._hierarchicalTime(fileTime)
            self.gmm_model = gmmModel
            self.timenum = timeNum
            self.timeClassifyLabel = timeClassifyLabel
        else:
            labels = self.test_knn_cluster(gmm, config["cluster_feature"])

        ucp_hash = self._generate_clus_dict(labels)
        uc_weight = self._generate_user_agent_num(ucp_hash)

        self.uc_weight = uc_weight

    def test_knn_cluster(self, gmm, cluster_feature):
        if cluster_feature == "all":
            fileTime = pd.DataFrame(
                self.data,
                columns=[
                    "pur_frequancy",
                    "order1_90",
                    "order2_90",
                    "order1_30",
                    "order2_30",
                    "order1_7",
                    "order2_7",
                    "order1_1",
                    "order2_1",
                    "tfa_year",
                    "tfa_month",
                    "tfa_day",
                    "tfa_weekday",
                    "tfa_weekday_1",
                    "tfa_weekday_2",
                    "tfa_weekday_3",
                    "tfa_weekday_4",
                    "tfa_weekday_5",
                    "tfa_weekday_6",
                    "tfa_weekday_7",
                    "tfa_season",
                    "tfa_season_0",
                    "tfa_season_1",
                    "tfa_season_2",
                    "tfa_season_3",
                ],
            )
            fileTime["tfa_year"] = fileTime["tfa_year"] - fileTime["tfa_year"].min()
        elif cluster_feature == "w-stat":
            fileTime = pd.DataFrame(
                self.data,
                columns=[
                    "tfa_year",
                    "tfa_month",
                    "tfa_day",
                    "tfa_weekday",
                    "tfa_weekday_1",
                    "tfa_weekday_2",
                    "tfa_weekday_3",
                    "tfa_weekday_4",
                    "tfa_weekday_5",
                    "tfa_weekday_6",
                    "tfa_weekday_7",
                    "tfa_season",
                    "tfa_season_0",
                    "tfa_season_1",
                    "tfa_season_2",
                    "tfa_season_3",
                ],
            )
            fileTime["tfa_year"] = fileTime["tfa_year"] - fileTime["tfa_year"].min()
        elif cluster_feature == "w-stru":
            fileTime = pd.DataFrame(
                self.data,
                columns=[
                    "pur_frequancy",
                    "order1_90",
                    "order2_90",
                    "order1_30",
                    "order2_30",
                    "order1_7",
                    "order2_7",
                    "order1_1",
                    "order2_1",
                ],
            )

        x = np.array(fileTime)
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        test_cluster_label = gmm.predict(x)

        return test_cluster_label

    def _get_all_cluster_feature(self, time2num):
        user_item_timestamp = np.array(self.user_item_timestamp)
        # =============================== Structural Features ================================= [90, 30, 7, 1]
        add_df = pd.DataFrame(
            columns=[
                "pur_frequancy",
                "order1_90",
                "order2_90",
                "order1_30",
                "order2_30",
                "order1_7",
                "order2_7",
                "order1_1",
                "order2_1",
            ],
            data=np.array([time2num[i] for i in user_item_timestamp]),
        )
        (
            self.data["pur_frequancy"],
            self.data["order1_90"],
            self.data["order2_90"],
            self.data["order1_30"],
            self.data["order2_30"],
            self.data["order1_7"],
            self.data["order2_7"],
            self.data["order1_1"],
            self.data["order2_1"],
        ) = (
            add_df["pur_frequancy"],
            add_df["order1_90"],
            add_df["order2_90"],
            add_df["order1_30"],
            add_df["order2_30"],
            add_df["order1_7"],
            add_df["order2_7"],
            add_df["order1_1"],
            add_df["order2_1"],
        )

        # =============================== Stastical Features =================================
        self.data["tfa_year"] = np.array([x.year for x in self.data.timestamps])
        self.data["tfa_month"] = np.array([x.month for x in self.data.timestamps])
        self.data["tfa_day"] = np.array([x.day for x in self.data.timestamps])
        self.data["tfa_weekday"] = np.array([x.isoweekday() for x in self.data.timestamps])

        tfa_weekday = pd.get_dummies(self.data.tfa_weekday, prefix="tfa_weekday")  # one hot encoding
        self.data = pd.concat((self.data, tfa_weekday), axis=1)

        self.data["tfa_season"] = np.array([self._get_season(x) for x in self.data.timestamps])
        tfa_season = pd.get_dummies(self.data.tfa_season, prefix="tfa_season")  # one hot encoding
        self.data = pd.concat((self.data, tfa_season), axis=1)

        fileTime = pd.DataFrame(
            self.data,
            columns=[
                "pur_frequancy",
                "order1_90",
                "order2_90",
                "order1_30",
                "order2_30",
                "order1_7",
                "order2_7",
                "order1_1",
                "order2_1",
                "tfa_year",
                "tfa_month",
                "tfa_day",
                "tfa_weekday",
                "tfa_weekday_1",
                "tfa_weekday_2",
                "tfa_weekday_3",
                "tfa_weekday_4",
                "tfa_weekday_5",
                "tfa_weekday_6",
                "tfa_weekday_7",
                "tfa_season",
                "tfa_season_0",
                "tfa_season_1",
                "tfa_season_2",
                "tfa_season_3",
            ],
        )
        fileTime["tfa_year"] = fileTime["tfa_year"] - fileTime["tfa_year"].min()

        return fileTime

    def _get_w_stat_cluster_feature(self):
        self.data["tfa_year"] = np.array([x.year for x in self.data.timestamps])
        self.data["tfa_month"] = np.array([x.month for x in self.data.timestamps])
        self.data["tfa_day"] = np.array([x.day for x in self.data.timestamps])
        self.data["tfa_weekday"] = np.array([x.isoweekday() for x in self.data.timestamps])

        tfa_weekday = pd.get_dummies(self.data.tfa_weekday, prefix="tfa_weekday")  # one hot encoding
        self.data = pd.concat((self.data, tfa_weekday), axis=1)

        self.data["tfa_season"] = np.array([self._get_season(x) for x in self.data.timestamps])
        tfa_season = pd.get_dummies(self.data.tfa_season, prefix="tfa_season")  # one hot encoding
        self.data = pd.concat((self.data, tfa_season), axis=1)

        fileTime = pd.DataFrame(
            self.data,
            columns=[
                "tfa_year",
                "tfa_month",
                "tfa_day",
                "tfa_weekday",
                "tfa_weekday_1",
                "tfa_weekday_2",
                "tfa_weekday_3",
                "tfa_weekday_4",
                "tfa_weekday_5",
                "tfa_weekday_6",
                "tfa_weekday_7",
                "tfa_season",
                "tfa_season_0",
                "tfa_season_1",
                "tfa_season_2",
                "tfa_season_3",
            ],
        )
        fileTime["tfa_year"] = fileTime["tfa_year"] - fileTime["tfa_year"].min()

        return fileTime

    def _get_w_stru_cluster_feature(self, time2num):
        add_df = pd.DataFrame(
            columns=[
                "pur_frequancy",
                "order1_90",
                "order2_90",
                "order1_30",
                "order2_30",
                "order1_7",
                "order2_7",
                "order1_1",
                "order2_1",
            ],
            data=np.array([time2num[i] for i in self.user_item_timestamp]),
        )
        (
            self.data["pur_frequancy"],
            self.data["order1_90"],
            self.data["order2_90"],
            self.data["order1_30"],
            self.data["order2_30"],
            self.data["order1_7"],
            self.data["order2_7"],
            self.data["order1_1"],
            self.data["order2_1"],
        ) = (
            add_df["pur_frequancy"],
            add_df["order1_90"],
            add_df["order2_90"],
            add_df["order1_30"],
            add_df["order2_30"],
            add_df["order1_7"],
            add_df["order2_7"],
            add_df["order1_1"],
            add_df["order2_1"],
        )

        fileTime = pd.DataFrame(
            self.data,
            columns=[
                "pur_frequancy",
                "order1_90",
                "order2_90",
                "order1_30",
                "order2_30",
                "order1_7",
                "order2_7",
                "order1_1",
                "order2_1",
            ],
        )

        return fileTime

    def _timeanalysis(self):
        dac_time = self.data.timestamps.value_counts()
        dac_time_date = pd.to_datetime(dac_time.index)

        dac_time_day = dac_time_date - dac_time_date.min()
        time2num = {}
        time2relative = {}
        serial2PrefixSum = {}
        for i in range(len(dac_time)):
            # quante volte occorrono i timestamp?
            time2num[dac_time.index[i]] = [dac_time.values[i]]
        for i in range(len(dac_time)):
            time2relative[dac_time_day.days[i]] = dac_time.index[i]
        mapIndex = sorted(time2relative.keys())
        serial2PrefixSum[0] = 0
        for i in range(1, mapIndex[-1] + 1):
            cur = time2num.get(time2relative.get(i, 0), 0)
            if cur:
                serial2PrefixSum[i] = serial2PrefixSum[i - 1] + cur[0]
            else:
                serial2PrefixSum[i] = serial2PrefixSum[i - 1]

        for gap in [90, 30, 7, 1]:
            self._structuralWithGap(gap, time2num, time2relative, mapIndex, serial2PrefixSum)
        return time2num

    def _hierarchicalTime(self, filetime):
        ui2label = {}
        x = np.array(filetime)
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

        models = GMM(self.config["cluster_num"], covariance_type="full", random_state=self.config["seed"]).fit(x)
        labels = models.predict(x)

        for user, item, label in zip(self.data.users, self.data.item, labels):
            ui2label[(user, item)] = label

        return models, self.config["cluster_num"], labels, ui2label

    def _generate_clus_dict(self, clus_label):
        uid_pid_clu = pd.DataFrame(self.data, columns=["users", "item"])
        uid_pid_clu["clu_label"] = clus_label

        uid_pid_clu_list = uid_pid_clu.values.tolist()
        ucp_hash = {}
        # depending on the server load this can take a long time
        for [uid, pid, clu] in progress_bar(uid_pid_clu_list, desc=f"generating clusters dict {self.set}"):
            if uid not in ucp_hash:
                # ucp_hash : {uids{c1: pid, c2:pid, ...}, ...}
                ucp_hash[uid] = {clu: [pid]}
            else:
                if clu not in ucp_hash[uid]:
                    ucp_hash[uid][clu] = []
                ucp_hash[uid][clu].append(pid)

        return ucp_hash

    def _generate_user_agent_num(self, ucp_hash):
        u_c_weight = ucp_hash
        # depending on the server load this can take a long time
        for u in progress_bar(u_c_weight, desc=f"generating user agent number {self.set}"):
            tmp_u_tot = 0
            for c in u_c_weight[u]:
                tmp_u_tot = tmp_u_tot + len(u_c_weight[u][c])
            for c in u_c_weight[u]:
                u_c_weight[u][c] = len(u_c_weight[u][c]) / tmp_u_tot
        return u_c_weight

    def _structuralWithGap(self, gap, time2Num, time2relative, mapIndex, serial2PrefixSum):
        # don't ask what this function does. I don't know.
        second_order_serial2PrefixSum = {}
        init_left = (serial2PrefixSum[2 * gap] - serial2PrefixSum[0] - 2 * serial2PrefixSum[gap]) / gap
        for i in range(mapIndex[-1] + 1):
            if i <= 2 * gap:
                second_order_serial2PrefixSum[i] = init_left
            else:
                second_order_serial2PrefixSum[i] = (
                    serial2PrefixSum[i] - 2 * serial2PrefixSum[i - gap] + serial2PrefixSum[i - 2 * gap]
                ) / gap

        init_left_2 = (
            second_order_serial2PrefixSum[2 * gap]
            - second_order_serial2PrefixSum[0]
            - 2 * second_order_serial2PrefixSum[gap]
        ) / gap
        for idx in mapIndex:
            if idx <= 2 * gap:
                gap_left = init_left
                gap_left_2 = init_left_2
            else:
                gap_left = (
                    serial2PrefixSum[idx] - 2 * serial2PrefixSum[idx - gap] + serial2PrefixSum[idx - 2 * gap]
                ) / gap
                gap_left_2 = (
                    second_order_serial2PrefixSum[idx]
                    - 2 * second_order_serial2PrefixSum[idx - gap]
                    + second_order_serial2PrefixSum[idx - 2 * gap]
                ) / gap

            time2Num[time2relative[idx]].append(gap_left)
            time2Num[time2relative[idx]].append(gap_left_2)

    def _get_season(self, dt):
        # dt = dt.date()
        dt = dt.replace(year=self.Y)
        return next(season for season, (start, end) in self.seasons if start <= dt <= end)


class TPRecDataset(KnowledgeBasedDataset):
    """
    A dataset class for temporal recommendation tasks, inheriting from :class:`KnowledgeBasedDataset`.
    This class is designed only to preprocess train, valid and test sets for temporal recommendation tasks.
    """

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        datasets = super().build()
        # Preprocess the datasets for temporal recommendation tasks
        train_set = datasets[0]  # train split
        valid_set = datasets[1]  # validation split
        test_set = datasets[2]  # test split

        # preprocess validation and test set and link them to train so we can use it in the model as attr
        train_set.temporal_weights = TPRecTimestampDataset(self.config, train_set.inter_feat, "train")
        valid_set.temporal_weights = TPRecTimestampDataset(
            self.config, valid_set.inter_feat, "validation", gmm=train_set.temporal_weights.gmm_model
        )
        test_set.temporal_weights = TPRecTimestampDataset(
            self.config, test_set.inter_feat, "test", gmm=train_set.temporal_weights.gmm_model
        )

        return datasets
