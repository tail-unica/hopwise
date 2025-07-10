# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

import random
import warnings
from itertools import chain, zip_longest

import numba
import numpy as np

from hopwise.data import Interaction
from hopwise.data.dataset import KnowledgeBasedDataset
from hopwise.data.utils import user_parallel_sampling
from hopwise.utils import PathLanguageModelingTokenType, progress_bar, set_color


class KnowledgePathDataset(KnowledgeBasedDataset):
    """:class:`KnowledgePathDataset` is based on :class:`~hopwise.data.dataset.KnowledgeBasedDataset`,
    and provides an interface to prepare tokenized knowledge graph path for path language modeling.

    Attributes:
        path_hop_length (int): The same as ``config["path_hop_length"]``.

        max_paths_per_user (int): The same as ``config["max_paths_per_user"]``.

        temporal_causality (bool): The same as ``config["path_sample_args"]["temporal_causality"]``.

        collaborative_path (bool): The same as ``config["path_sample_args"]["collaborative_path"]``.
        Not used when :attr:`strategy` = `metapaths` as collaborative metapaths must be explicitly defined.

        strategy (str): The same as ``config["path_sample_args"]["strategy"]``.

        path_token_separator (str): The same as ``config["path_sample_args"]["path_token_separator"]``.

        restrict_by_phase (bool): The same as ``config["path_sample_args"]["restrict_by_phase"]``.

        max_consecutive_invalid (int): The same as ``config["MAX_CONSECUTIVE_INVALID"]``.

        tokenizer (PreTrainedTokenizerFast): Tokenizer to process the sample paths.
    """

    PATH_PADDING = -1

    def __init__(self, config):
        super().__init__(config)
        self._path_dataset = None  # path dataset is generated with generate_user_path_dataset
        self._tokenized_dataset = None  # tokenized path dataset is generated with tokenize_path_dataset
        self._tokenizer = None
        self.used_ids = None

        self._init_tokenizer()

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.context_length = self.config["context_length"]

        # Path sampling parameters
        self.path_hop_length = self.config["path_hop_length"]
        assert self.path_hop_length % 2 == 1, "Path hop length must be odd"
        self.max_paths_per_user = self.config["MAX_PATHS_PER_USER"]

        # path_hop_length = n_relations => (n_relations + user_starting_node) + n_relations + 2 (BOS, EOS)
        self.token_sequence_length = (1 + self.path_hop_length) + self.path_hop_length + 2

        path_sample_args = self.config["path_sample_args"]
        self.temporal_causality = path_sample_args["temporal_causality"]
        self.collaborative_path = path_sample_args["collaborative_path"]
        self.strategy = path_sample_args["strategy"]
        self.path_token_separator = path_sample_args["path_token_separator"]
        self.restrict_by_phase = path_sample_args["restrict_by_phase"]
        self.max_consecutive_invalid = path_sample_args["MAX_CONSECUTIVE_INVALID"]
        self.parallel_max_workers = path_sample_args["parallel_max_workers"]

        # Tokenizer parameters
        self.tokenizer_model = self.config["tokenizer"]["model"]

        # Special tokens
        if self.config["tokenizer"]["special_tokens"] is not None:
            for token_name, token_value in self.config["tokenizer"]["special_tokens"].items():
                setattr(self, token_name, token_value)
            self.special_tokens = list(self.config["tokenizer"]["special_tokens"].values())
        else:
            self.special_tokens = []

        self.logger.debug(set_color("tokenizer", "blue") + f": {self.tokenizer_model}")

    @property
    def path_dataset(self):
        if self._path_dataset is None:
            raise ValueError("Path dataset has not been generated yet, build the dataset first.")

        return self._path_dataset

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenized_dataset(self):
        if self._tokenized_dataset is None:
            raise ValueError("Tokenized path dataset has not been generated yet, build the dataset first.")

        return self._tokenized_dataset

    def __len__(self):
        """Return the length of the tokenized dataset."""
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        """Return the item at index `idx` from the tokenized dataset."""
        if self._tokenized_dataset is None:
            # It avoids issues with hopwise flops calculation.
            dummy_data = self.tokenize(["U1"])
            return Interaction(dummy_data.data)

        return self.tokenized_dataset[idx]

    def _init_tokenizer(self):
        """Initialize the HuggingFace tokenizer."""
        from tokenizers import Tokenizer, pre_tokenizers
        from tokenizers import models as token_models
        from tokenizers import processors as token_processors
        from tokenizers import trainers as token_trainers
        from transformers import PreTrainedTokenizerFast

        tokenizer_model_class = getattr(token_models, self.tokenizer_model)

        tokenizer_object = Tokenizer(tokenizer_model_class(unk_token=self.unk_token))

        # Pre-tokenizer definition based on :attr:`path_token_separator`
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(self.path_token_separator, "removed")

        entity_range = np.arange(self.item_num, self.entity_num)  # only entities that are not items are considered
        token_vocab = np.concatenate(
            [
                np.char.add(PathLanguageModelingTokenType.USER.token, np.arange(self.user_num).astype(str)),
                np.char.add(PathLanguageModelingTokenType.ITEM.token, np.arange(self.item_num).astype(str)),
                np.char.add(PathLanguageModelingTokenType.ENTITY.token, entity_range.astype(str)),
                np.char.add(PathLanguageModelingTokenType.RELATION.token, np.arange(self.relation_num).astype(str)),
            ]
        )

        tokenizer_trainer_class = getattr(token_trainers, self.tokenizer_model + "Trainer")
        tokenizer_trainer = tokenizer_trainer_class(
            vocab_size=len(token_vocab) + len(self.special_tokens), special_tokens=self.special_tokens
        )

        tokenizer_object.train_from_iterator(token_vocab, trainer=tokenizer_trainer)

        tokenizer_object.post_processor = token_processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (spec_token, tokenizer_object.token_to_id(spec_token))
                for spec_token in [self.bos_token, self.eos_token]
            ],
        )

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            model_max_length=self.context_length,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
        )

    def get_tokenized_ckg(self):
        """Return the tokenized collaborative knowledge graph.

        We assume the any path is bidirectional except for user-item relations and :attr:`collaborative_path` is False.

        Returns:
            dict[dict[set]]: The tokenized collaborative knowledge graph.
        """
        token_vocab = self.tokenizer.get_vocab()
        graph = self._create_ckg_igraph(show_relation=True, directed=False)
        vertex_metadata, edge_metadata = graph.to_dict_list()

        def igraph_id_to_tokenizer_id(igraph_head, igraph_relation, igraph_tail):
            ret = []
            triple = [igraph_head, igraph_relation, igraph_tail]
            for term, term_type in zip(triple, ["node", "relation", "node"]):
                term_id = term
                if term_type == "node":
                    if vertex_metadata[term_id]["type"] == self.uid_field:
                        prefix = PathLanguageModelingTokenType.USER.token
                    elif vertex_metadata[term_id]["type"] == self.iid_field:
                        term_id -= self.user_num
                        prefix = PathLanguageModelingTokenType.ITEM.token
                    elif vertex_metadata[term_id]["type"] == self.entity_field:
                        prefix = PathLanguageModelingTokenType.ENTITY.token
                        term_id -= self.user_num
                    else:
                        raise ValueError(
                            f"Unknown vertex type [{vertex_metadata[term_id]['type']}] "
                            "in igraph during tokenized_kg generation."
                        )
                else:
                    prefix = PathLanguageModelingTokenType.RELATION.token

                token_id = token_vocab[prefix + str(term_id)]
                ret.append(token_id)

            return ret

        tokenized_kg = {}
        for edge in edge_metadata:
            head = edge["source"]
            tail = edge["target"]
            relation = edge["type"]
            relation_id = self.field2token_id[self.relation_field][relation]

            head_token, relation_token, tail_token = igraph_id_to_tokenizer_id(head, relation_id, tail)

            # head is always the user in user-item relations. The check to add the reverse path is done later
            if relation == self.ui_relation and vertex_metadata[head]["type"] != self.uid_field:
                head_token, tail_token = tail_token, head_token

            if head_token not in tokenized_kg:
                tokenized_kg[head_token] = {}
            if tail_token not in tokenized_kg:
                tokenized_kg[tail_token] = {}

            if relation_token not in tokenized_kg[head_token]:
                tokenized_kg[head_token][relation_token] = set()

            tokenized_kg[head_token][relation_token].add(tail_token)

            if relation_token not in tokenized_kg[tail_token]:
                tokenized_kg[tail_token][relation_token] = set()

            tokenized_kg[tail_token][relation_token].add(head_token)

        return tokenized_kg

    def tokenize(self, data):
        """Tokenize the input data using the tokenizer."""
        return self.tokenizer(
            data,
            truncation=True,
            padding=True,
            max_length=self.context_length,
            add_special_tokens=True,
        )

    def tokenize_path_dataset(self):
        """Tokenize the path dataset."""

        if self._tokenized_dataset is None:
            tokenized_dataset = self.tokenize(self.path_dataset.split("\n"))
            tokenized_dataset = Interaction(tokenized_dataset.data)
            correct_path_mask = [
                all(spec_token not in path[1:-1] for spec_token in self.tokenizer.all_special_ids)
                for path in tokenized_dataset["input_ids"]
            ]
            tokenized_dataset = tokenized_dataset[correct_path_mask]
            self._tokenized_dataset = tokenized_dataset

    def build(self):
        """Extends the build method to generate user path dataset and tokenize it."""
        datasets = super().build()
        datasets[0].generate_user_path_dataset()
        datasets[0].tokenize_path_dataset()

        return datasets

    def get_tokenized_used_ids(self):
        """Convert the used ids to tokenized ids.

        Args:
            used_ids: A numpy array of sets, where each set contains the item ids
            that a user has interacted with.
            tokenizer: The tokenizer to convert ids to tokenized ids.
        Returns:
            dict: A dictionary where keys are tokenized user ids and values are sets of tokenized item ids.
                A numpy array of sets cannot be used as user tokens are not in the range [0, user_num].
        """
        user_token_type = PathLanguageModelingTokenType.USER.token
        item_token_type = PathLanguageModelingTokenType.ITEM.token

        used_ids = self.get_user_used_ids()
        tokenized_used_ids = {}
        for uid in range(used_ids.shape[0]):
            uid_token = self.tokenizer.convert_tokens_to_ids(user_token_type + str(uid))
            tokenized_used_ids[uid_token] = set(
                [self.tokenizer.convert_tokens_to_ids(item_token_type + str(item)) for item in used_ids[uid]]
            )
        return tokenized_used_ids

    def generate_user_path_dataset(self):
        """Generate path dataset by sampling paths from the knowledge graph.

        Paths represent walks in the graph that connect :attr:`hop_length` + 1 entities through
        :attr:`hop_length` relations.
        Each path connects two items. In the common scenario, the first item is a positive item
        for the user and the second item is a recommendation candidate.

        Refer to :meth:`generate_user_paths` for more details about path generation strategies.
        """
        if not isinstance(self.inter_feat, Interaction):
            raise ValueError("The data should be prepared before generating the path dataset.")

        if self._path_dataset is None:
            generated_paths = self.generate_user_paths()

            path_string = ""
            for path in generated_paths:
                path_string += self._format_path(path) + "\n"
            self._path_dataset = path_string

    def generate_user_paths(self):
        """Generate paths from the knowledge graph.

        It currently supports four sampling strategies:

        - weighted-rw: sampling-and-discarding approach through weighted random walk.
                       Paths are sampled from the knowledge graph and discarded if they are not valid, i.e.,
                       they do not end in a positive item

        - constrained-rw: faithful random walk with constraints based on expected path output.

        - simple: randomly sample a positive item for each user and extract all simple paths to other positive items.

        - simple-ui: randomly sample a positive item for each user and extract all simple paths to all positive items.

        - metapath: random walk constrained by pre-defined metapaths.

        Returns:
            list: List of paths with relations.
        """
        if self.strategy in ["weighted-rw", "constrained-rw", "simple", "simple-ui"]:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
        elif self.strategy in ["metapath"]:
            graph = self.ckg_hetero_graph(form="dgl", directed=not self.collaborative_path)
        else:
            raise NotImplementedError(f"Path generation method [{self.strategy}] has not been implemented.")

        temporal_matrix = None
        if self.temporal_causality:
            if self.time_field in self.inter_feat:
                temporal_matrix = self.inter_matrix(value_field=self.time_field).toarray()
            else:
                self.logger.warning(
                    "time_field has not been loaded or set,"
                    "thus temporal causality will not be used for path generation."
                )

        used_ids = self.get_user_used_ids()

        if self.strategy == "weighted-rw":
            if not self.collaborative_path:
                # prevent ui-relations to be sampled
                kg_rel_num = len(self.relations)
                graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num
            else:
                graph.es["weight"] = [1.0] * graph.ecount()

            max_tries_per_iid = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]
            generated_paths = self._generate_user_paths_weighted_random_walk(
                graph, used_ids, temporal_matrix=temporal_matrix, max_tries_per_iid=max_tries_per_iid
            )
        elif self.strategy == "constrained-rw":
            max_paths_per_hop = self.config["path_sample_args"]["MAX_RW_PATHS_PER_HOP"]
            generated_paths = self._generate_user_paths_constrained_random_walk(
                graph, used_ids, temporal_matrix=temporal_matrix, paths_per_hop=max_paths_per_hop
            )
        elif self.strategy == "simple":
            generated_paths = self._generate_user_paths_all_simple(graph, used_ids, temporal_matrix=temporal_matrix)
        elif self.strategy == "simple-ui":
            generated_paths = self._generate_user_paths_all_simple_ui(graph, used_ids, temporal_matrix=temporal_matrix)
        elif self.strategy == "metapath":
            generated_paths = self._generate_user_paths_from_metapaths(
                graph, used_ids, temporal_matrix=temporal_matrix
            )
        else:
            raise NotImplementedError(f"Path generation method [{self.strategy}] has not been implemented.")

        if self.strategy != "metapath":
            paths_with_relations = self._add_paths_relations(graph, generated_paths)
        else:
            padded_generated_paths = list(zip_longest(*generated_paths, fillvalue=self.PATH_PADDING))
            paths_with_relations = np.array(padded_generated_paths).T

        return paths_with_relations

    def _generate_user_paths_weighted_random_walk(self, graph, used_ids, temporal_matrix=None, max_tries_per_iid=50):
        """Generate paths from the knowledge graph using weighted random walk.

        The last hop is not sampled, but it is selected according to the item candidates from temporal matrix
        if a relation between the second to last node and the item candidates exists.
        """
        paths = set()

        kwargs = dict(
            parallel_max_workers=self.parallel_max_workers,
            temporal_matrix=temporal_matrix,
            max_tries_per_iid=max_tries_per_iid,
            path_hop_length=self.path_hop_length - 2,
            user_num=self.user_num,
            item_num=self.item_num,
            max_consecutive_invalid=self.max_consecutive_invalid,
            max_paths_per_user=self.max_paths_per_user,
            restrict_by_phase=self.restrict_by_phase,
            collaborative_path=self.collaborative_path,
        )

        user_paths = _generate_user_paths_weighted_random_walk_per_user(graph, used_ids, self.iid_field, **kwargs)
        paths = set.union(*user_paths)

        return paths

    def _generate_user_paths_constrained_random_walk(self, graph, used_ids, temporal_matrix=None, paths_per_hop=1):
        """Generate paths from the knowledge graph using constrained random walks, similar to DGL random walk based on
        metapaths (https://docs.dgl.ai/en/1.1.x/generated/dgl.sampling.random_walk.html).

        The difference is that this strategy is constrained to the knowledge graph relations, but not to pre-defined
        metapahts. Then, the resulting paths may not be semantically sound, but they are still valid.

        Args:
            paths_per_hop (int, optional): The number of paths sampled at each hop to continue the random walk.
            Defaults to 1.
        """
        paths = set()
        kwargs = dict(
            parallel_max_workers=self.parallel_max_workers,
            temporal_matrix=temporal_matrix,
            paths_per_hop=paths_per_hop,
            path_hop_length=self.path_hop_length - 1,
            user_num=self.user_num,
            max_consecutive_invalid=self.max_consecutive_invalid,
            max_paths_per_user=self.max_paths_per_user,
            restrict_by_phase=self.restrict_by_phase,
            collaborative_path=self.collaborative_path,
        )

        user_paths = _generate_user_paths_constrained_random_walk_per_user(
            graph, used_ids, self.iid_field, self.entity_field, **kwargs
        )
        paths = set.union(*user_paths)

        return paths

    def _generate_user_paths_all_simple(self, graph, used_ids, temporal_matrix=None):
        """Generate paths from the knowledge graph by extracting all simple paths for a randomly sampled item.
        Refer to igraph's https://python.igraph.org/en/stable/api/igraph.Graph.html#get_all_simple_paths.

        It considers all valid simple paths between each user and randomly sampled item,
        so resulting paths are not much diverse, as not all starting <user, item> pairs might be sampled.
        """
        paths = set()

        kwargs = dict(
            parallel_max_workers=self.parallel_max_workers,
            temporal_matrix=temporal_matrix,
            path_hop_length=self.path_hop_length - 1,
            user_num=self.user_num,
            item_num=self.item_num,
            max_paths_per_user=self.max_paths_per_user,
            restrict_by_phase=self.restrict_by_phase,
            collaborative_path=self.collaborative_path,
        )

        user_paths = _generate_user_paths_all_simple_per_user(graph, used_ids, **kwargs)
        paths = set.union(*user_paths)

        return paths

    def _generate_user_paths_all_simple_ui(self, graph, used_ids, temporal_matrix=None):
        """Generate paths from the knowledge graph by extracting all simple paths for all the positives..
        Refer to igraph's https://python.igraph.org/en/stable/api/igraph.Graph.html#get_all_simple_paths.

        It sample all the paths from a user to all the positive items. If U1 has 3 positive items, we'd have
        k distinct paths from U1 to each of the positive items.
        """
        paths = set()

        kwargs = dict(
            parallel_max_workers=self.parallel_max_workers,
            temporal_matrix=temporal_matrix,
            path_hop_length=self.path_hop_length - 1,
            user_num=self.user_num,
            item_num=self.item_num,
            max_paths_per_user=self.max_paths_per_user,
            collaborative_path=self.collaborative_path,
        )

        user_paths = _generate_user_paths_all_simple_per_user_and_positive(graph, used_ids, **kwargs)
        paths = set.union(*user_paths)
        return paths

    def _generate_user_paths_from_metapaths(self, graph, used_ids, temporal_matrix=None):
        """Generate paths from pre-defined metapaths. Refer to DGL's random walk based on metapaths
        https://docs.dgl.ai/en/1.1.x/generated/dgl.sampling.random_walk.html for more details.
        """
        import dgl
        import torch

        final_paths = set()

        iter_users = progress_bar(
            range(1, self.user_num),
            total=self.user_num - 1,
            ncols=100,
            desc=set_color("KG Path Sampling", "red", progress=True),
        )

        if temporal_matrix is not None:
            temporal_matrix = torch.from_numpy(temporal_matrix)

        # Filter metapaths that do not match the hop length
        base_metapaths = self.config["metapaths"]
        # metapaths = list(filter(lambda mp: len(mp) == path_hop_length, metapaths))
        metapaths = np.empty(len(base_metapaths), dtype=object)
        metapaths[:] = base_metapaths
        for u in iter_users:
            pos_iid = torch.tensor(list(used_ids[u]))
            if temporal_matrix is not None:
                pos_iid = pos_iid[torch.argsort(temporal_matrix[u, pos_iid])]

            user_path_sample_size = 0
            user_invalid_paths = self.max_consecutive_invalid
            while True:
                # select new starting node. If temporal last pos item can only be at the end of the path
                start_nodes = pos_iid if temporal_matrix is None else pos_iid[:-1]

                generated_path_nodes, relations, node_types = [], [], []
                # First hop is the relation user-item already addressed
                for mp in metapaths[np.random.permutation(len(metapaths))]:
                    try:
                        mp_nodes, mp_types = dgl.sampling.random_walk(graph, start_nodes, metapath=mp)
                    except dgl._ffi.base.DGLError as error:
                        error.args = (f"The metapath {mp} raised the error [{error.args[0].lower()}]",)
                        raise (error)

                    generated_path_nodes.append(mp_nodes)
                    mp_types = mp_types.unsqueeze(0)
                    mp_types = mp_types.expand(mp_nodes.shape[0], -1)
                    node_types.append(mp_types)

                    relation_map = self.field2token_id[self.relation_field]
                    if isinstance(mp[0], tuple):
                        mp_with_ui_rel = [(self.uid_field, self.ui_relation, self.iid_field), *mp]
                        mp_relations = torch.Tensor([relation_map[mp_tuple[1]] for mp_tuple in mp_with_ui_rel])
                    else:
                        mp_with_ui_rel = [self.ui_relation, *mp]
                        mp_relations = torch.Tensor([relation_map[rel] for rel in mp_with_ui_rel])
                    mp_relations = mp_relations.unsqueeze(0)
                    mp_relations = mp_relations.expand(mp_nodes.shape[0], -1)
                    relations.append(mp_relations)

                def filter_and_validate_metapaths(pnodes, rels, ntypes):
                    nonlocal user_path_sample_size
                    nonlocal user_invalid_paths

                    pnodes = torch.vstack(pnodes)
                    rels = torch.vstack(rels)
                    ntypes = torch.vstack(ntypes)
                    path_hop_length = pnodes.shape[1]

                    # filter valid random walks
                    valid_path_node_mask = ~(pnodes == -1).any(dim=1)
                    pnodes = pnodes[valid_path_node_mask]
                    rels = rels[valid_path_node_mask]
                    ntypes = ntypes[valid_path_node_mask]

                    if self.restrict_by_phase:
                        # filter paths that do not end in a positive item
                        pos_iid_mask = torch.full((self.item_num,), fill_value=-1, dtype=int)
                        pos_iid_mask[pos_iid] = torch.arange(pos_iid.shape[0])
                        start_end_nodes = pnodes[:, [0, -1]]
                        start_end_nodes_pos_idxs = pos_iid_mask[start_end_nodes]
                        valid_path_node_mask = ~(start_end_nodes_pos_idxs == -1).any(dim=1)
                        if temporal_matrix is not None:
                            pos_idxs_check = start_end_nodes_pos_idxs[:, 1] > start_end_nodes_pos_idxs[:, 0]
                            valid_path_node_mask = torch.logical_and(valid_path_node_mask, pos_idxs_check)
                        else:
                            pos_idxs_check = start_end_nodes_pos_idxs[:, 0] != start_end_nodes_pos_idxs[:, 1]
                            valid_path_node_mask = torch.logical_and(valid_path_node_mask, pos_idxs_check)
                    else:
                        valid_path_node_mask = pnodes[:, 0] != pnodes[:, -1]
                    valid_path_nodes = pnodes[valid_path_node_mask]
                    valid_relations = rels[valid_path_node_mask]
                    valid_node_types = ntypes[valid_path_node_mask]

                    if valid_path_nodes.shape[0] > 0:
                        # remap entities to dataset ids
                        entity_idx = graph.ntypes.index(self.entity_field)
                        paths_entities_map = valid_node_types == entity_idx
                        valid_path_nodes[paths_entities_map] += self.item_num

                        # remap non-user entities ids to homogeneous ids (entity ids after item ids after user ids)
                        non_user_idx = graph.ntypes.index(self.uid_field)
                        paths_non_users_map = valid_node_types != non_user_idx
                        valid_path_nodes[paths_non_users_map] += self.user_num

                        paths_with_relations = torch.zeros(
                            (valid_path_nodes.shape[0], path_hop_length * 2 + 1), dtype=int
                        )
                        paths_with_relations[:, 0] = u
                        paths_with_relations[:, 1::2] = valid_relations
                        paths_with_relations[:, 2::2] = valid_path_nodes
                        paths_with_relations = paths_with_relations.unique(dim=0)

                        n_paths = min(self.max_paths_per_user - user_path_sample_size, paths_with_relations.shape[0])
                        paths_with_relations = paths_with_relations[:n_paths]

                        user_path_sample_size += paths_with_relations.shape[0]
                        final_paths.update(map(tuple, paths_with_relations.numpy().tolist()))

                        user_invalid_paths = self.max_consecutive_invalid
                    else:
                        user_invalid_paths -= 1

                # Group a list of torch tensors based on the second dimension to speed-up path filtering and validation
                path_length_groups = {}
                for paths_mp_i in range(len(generated_path_nodes)):
                    path_length = generated_path_nodes[paths_mp_i].shape[1]
                    if path_length not in path_length_groups:
                        path_length_groups[path_length] = {"path_nodes": [], "relations": [], "node_types": []}
                    path_length_groups[path_length]["path_nodes"].append(generated_path_nodes[paths_mp_i])
                    path_length_groups[path_length]["relations"].append(relations[paths_mp_i])
                    path_length_groups[path_length]["node_types"].append(node_types[paths_mp_i])

                for path_length_gr in path_length_groups.values():
                    filter_and_validate_metapaths(
                        path_length_gr["path_nodes"], path_length_gr["relations"], path_length_gr["node_types"]
                    )

                if user_path_sample_size == self.max_paths_per_user or user_invalid_paths == 0:
                    break
        return final_paths

    @staticmethod
    def _check_kg_path(path, user_num, item_num, check_last_node=False, collaborative_path=False):
        """Check if the path is valid. The first node must be an item node and it assumes the user node is omitted.

        Args:
            path (list): The path to be checked.

            check_last_node (bool, optional): Whether to check the last node in the path.
            Defaults to ``False``.
        """
        path = np.array(path, dtype=int)
        graph_min_iid = 1 + user_num
        graph_max_iid = item_num - 1 + user_num

        user_check = path[0] < graph_min_iid
        pos_iid_check = graph_min_iid <= path[1] <= graph_max_iid
        valid_path = (path[2:-1] >= graph_min_iid).all() or collaborative_path
        check_rec_iid = not check_last_node or graph_min_iid <= path[-1] <= graph_max_iid

        return user_check and pos_iid_check and valid_path and check_rec_iid

    def _format_path(self, path):
        """Format the path to a string according to :class:`~hopwise.utils.enum_type.PathLanguageModelingTokenType`.

        Args:
            path (list): The path to be formatted.
        """
        path = path[path != self.PATH_PADDING]  # remove padding for shorter paths
        path_nodes = path[::2]
        path_relations = path[1::2]

        remapped_path_nodes = []
        graph_min_iid = self.user_num
        graph_max_iid = self.item_num - 1 + self.user_num
        for node in path_nodes:
            if graph_min_iid <= node <= graph_max_iid:
                remapped_path_nodes.append(PathLanguageModelingTokenType.ITEM.token + str(node - self.user_num))
            elif node < graph_min_iid:
                remapped_path_nodes.append(PathLanguageModelingTokenType.USER.token + str(node))
            else:
                remapped_path_nodes.append(PathLanguageModelingTokenType.ENTITY.token + str(node - self.user_num))

        relation_mapped_list = [PathLanguageModelingTokenType.RELATION.token + str(r) for r in path_relations]

        interleaved_entities_relations = zip_longest(remapped_path_nodes, relation_mapped_list)
        path_string = self.path_token_separator.join(list(chain(*interleaved_entities_relations))[:-1])

        return path_string

    def _add_paths_relations(self, graph, paths):
        n_paths = len(paths)
        paths_array = np.full((n_paths, self.path_hop_length + 1), fill_value=self.PATH_PADDING, dtype=int)
        for i, path in enumerate(paths):
            paths_array[i, : len(path)] = path

        complete_path_length = self.path_hop_length * 2 + 1
        paths_with_relations = np.full((n_paths, complete_path_length), fill_value=self.PATH_PADDING, dtype=int)
        relation_token_id = self.field2token_id[self.relation_field]
        relation_map = np.zeros((len(graph.vs), len(graph.vs)), dtype=int)
        for edge in graph.es:
            relation_map[edge.source, edge.target] = relation_token_id[edge["type"]]
            if not graph.is_directed():
                relation_map[edge.target, edge.source] = relation_token_id[edge["type"]]

        _add_paths_relations_parallel(paths_array, paths_with_relations, relation_map)

        return paths_with_relations

    def __str__(self):
        info = [
            super().__str__(),
            f"The number of hops used for path sampling: {self.path_hop_length}",
            f"Maximum number of paths sampled per user: {self.max_paths_per_user}",
            f"The path sampling strategy: {self.strategy}",
            f"The tokenizer model: {self.tokenizer_model}",
        ]
        return "\n".join(info)


def _check_temporal_causality_feasibility(temporal_matrix, pos_iid):
    """Check if temporal causality is feasible for the given positive item ids."""
    temporal_start = int(temporal_matrix is not None)
    if pos_iid.shape[0] - temporal_start <= 0:
        warnings.warn(
            "Some users have only one positive item, thus path sampling with temporal causality is not feasible. "
            "The current user will be skipped.",
            RuntimeWarning,
        )
        return None
    return pos_iid.shape[0] - temporal_start


@user_parallel_sampling
def _generate_user_paths_constrained_random_walk_per_user(graph, used_ids, iid_field, entity_field, **kwargs):
    """Parallel version of the constrained random walk path generation."""
    temporal_matrix = kwargs.pop("temporal_matrix", None)
    paths_per_hop = kwargs.pop("paths_per_hop", None)
    path_hop_length = kwargs.pop("path_hop_length", None)
    user_num = kwargs.pop("user_num", None)
    max_consecutive_invalid = kwargs.pop("max_consecutive_invalid", None)
    max_paths_per_user = kwargs.pop("max_paths_per_user", None)
    restrict_by_phase = kwargs.pop("restrict_by_phase", None)
    collaborative_path = kwargs.pop("collaborative_path", None)

    def process_user(u):
        user_paths = set()

        pos_iid = np.array(list(used_ids[u]))
        if temporal_matrix is not None:
            pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

        # reindex item ids according to the igraph
        pos_iid += user_num

        user_path_sample_size = 0
        user_invalid_paths = max_consecutive_invalid

        def _graph_traversal(g, path, hop, candidates=None):
            nonlocal user_paths
            nonlocal user_path_sample_size
            nonlocal user_invalid_paths

            if hop == 1 and candidates is not None:
                next_node_candidates = g.es.select(_source=path[-1], _target=candidates)
                next_node_candidates = list(
                    set(e.source if e.source_vertex != path[-1] else e.target for e in next_node_candidates)
                )
            else:

                def _check_next_candidate(node):
                    if hop == 1:
                        type_check = g.vs[node]["type"] == iid_field
                    elif collaborative_path:
                        type_check = g.vs[node]["type"] != iid_field
                    else:
                        type_check = g.vs[node]["type"] == entity_field

                    return type_check and node != path[-1]

                next_node_candidates = [v for v in g.neighbors(path[-1]) if _check_next_candidate(v)]

            next_nodes = np.random.choice(
                next_node_candidates, min(len(next_node_candidates), paths_per_hop), replace=False
            )
            for node in next_nodes:
                new_path = (*path, node)
                if hop == 1:
                    # Path is valid per construction
                    user_paths.add(new_path)
                    user_path_sample_size += 1
                else:
                    _graph_traversal(g, new_path, hop - 1, candidates)

                if user_path_sample_size == max_paths_per_user:
                    return

        while True:
            pos_iid_range = _check_temporal_causality_feasibility(temporal_matrix, pos_iid)
            if pos_iid_range is None:
                return set()

            # select new starting node
            start_node_idx = np.random.randint(pos_iid_range)
            start_node = pos_iid[start_node_idx]

            if restrict_by_phase:
                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])
            else:
                item_candidates = None

            # First hop is the relation user-item already addressed
            curr_path_sample_size = user_path_sample_size
            _graph_traversal(graph, (u, start_node), path_hop_length, item_candidates)
            if user_path_sample_size - curr_path_sample_size == 0:
                user_invalid_paths -= paths_per_hop
            else:
                user_invalid_paths = max_consecutive_invalid

            if user_path_sample_size == max_paths_per_user or user_invalid_paths <= 0:
                break

        return user_paths

    return process_user


@user_parallel_sampling
def _generate_user_paths_weighted_random_walk_per_user(graph, used_ids, iid_field, **kwargs):
    """Parallel version of the weighted random walk path generation."""
    temporal_matrix = kwargs.pop("temporal_matrix", None)
    max_tries_per_iid = kwargs.pop("max_tries_per_iid", None)
    path_hop_length = kwargs.pop("path_hop_length", None)
    user_num = kwargs.pop("user_num", None)
    item_num = kwargs.pop("item_num", None)
    max_consecutive_invalid = kwargs.pop("max_consecutive_invalid", None)
    max_paths_per_user = kwargs.pop("max_paths_per_user", None)
    restrict_by_phase = kwargs.pop("restrict_by_phase", None)
    collaborative_path = kwargs.pop("collaborative_path", None)

    def process_user(u):
        user_paths = set()

        pos_iid = np.array(list(used_ids[u]))
        if temporal_matrix is not None:
            pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

        # reindex item ids according to the igraph
        pos_iid += user_num

        iid_tries = 0
        user_path_sample_size = 0
        user_invalid_paths = max_consecutive_invalid
        while True:
            if iid_tries == 0:
                iid_tries = max_tries_per_iid

                pos_iid_range = _check_temporal_causality_feasibility(temporal_matrix, pos_iid)
                if pos_iid_range is None:
                    return set()

                # select new starting node
                start_node_idx = np.random.randint(pos_iid_range)
                start_node = pos_iid[start_node_idx]

            if restrict_by_phase:
                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])
            else:
                item_candidates = None

            while iid_tries > 0:
                # First hop is the relation user-item already addressed
                generated_path = graph.random_walk(start_node, path_hop_length, weights="weight")
                full_path = (u, *generated_path)

                valid_path = KnowledgePathDataset._check_kg_path(
                    (*full_path, -1), user_num, item_num, check_last_node=False, collaborative_path=collaborative_path
                )
                if not valid_path:
                    iid_tries -= 1
                else:
                    break

            if valid_path:
                if item_candidates is not None:
                    reachable_candidates = graph.es.select(_source=full_path[-1], _target=item_candidates)
                    reachable_candidates = set(
                        e.source if e.source_vertex["type"] == iid_field else e.target for e in reachable_candidates
                    )
                    reachable_candidates = set([e for e in reachable_candidates if e != start_node])
                else:
                    reachable_candidates = [
                        v
                        for v in graph.neighbors(full_path[-1])
                        if graph.vs[v]["type"] == iid_field and v != start_node
                    ]

                if len(reachable_candidates) > 0:
                    last_node = np.random.choice(list(reachable_candidates))
                    full_path = (*full_path, last_node)
                else:
                    valid_path = False

            if valid_path:
                if full_path not in user_paths:
                    user_paths.add(full_path)
                    user_path_sample_size += 1
                    user_invalid_paths = max_consecutive_invalid
                else:
                    user_invalid_paths -= 1
            else:
                user_invalid_paths -= 1

            if user_path_sample_size == max_paths_per_user or user_invalid_paths == 0:
                break

        return user_paths

    return process_user


@user_parallel_sampling
def _generate_user_paths_all_simple_per_user_and_positive(graph, used_ids, **kwargs):
    """Parallel version of the simple path generation."""
    temporal_matrix = kwargs.pop("temporal_matrix", None)
    path_hop_length = kwargs.pop("path_hop_length", None)
    user_num = kwargs.pop("user_num", None)
    item_num = kwargs.pop("item_num", None)
    max_paths_per_user = kwargs.pop("max_paths_per_user", None)
    collaborative_path = kwargs.pop("collaborative_path", None)

    def process_user(u):
        user_paths = set()

        pos_iid = np.array(list(used_ids[u]))
        if temporal_matrix is not None:
            pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

        # reindex item ids according to the igraph
        pos_iid += user_num

        for target_item in pos_iid:
            user_path_sample_size = 0

            # First hop is the relation user-item already addressed
            generated_paths = graph.get_all_simple_paths(u, to=target_item, cutoff=path_hop_length + 1, mode="all")

            random.shuffle(generated_paths)
            # U R I R I R I
            for full_path in generated_paths:
                valid_path = KnowledgePathDataset._check_kg_path(
                    full_path, user_num, item_num, check_last_node=True, collaborative_path=collaborative_path
                )

                if valid_path not in user_paths:
                    user_paths.add(tuple(full_path))
                    user_path_sample_size += 1

                if user_path_sample_size == max_paths_per_user:
                    break

        return user_paths

    return process_user


@user_parallel_sampling
def _generate_user_paths_all_simple_per_user(graph, used_ids, **kwargs):
    """Parallel version of the simple path generation."""
    temporal_matrix = kwargs.pop("temporal_matrix", None)
    path_hop_length = kwargs.pop("path_hop_length", None)
    user_num = kwargs.pop("user_num", None)
    item_num = kwargs.pop("item_num", None)
    max_paths_per_user = kwargs.pop("max_paths_per_user", None)
    restrict_by_phase = kwargs.pop("restrict_by_phase", None)
    collaborative_path = kwargs.pop("collaborative_path", None)

    all_items = np.arange(1, item_num) + user_num

    def process_user(u):
        user_paths = set()

        pos_iid = np.array(list(used_ids[u]))
        if temporal_matrix is not None:
            pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

        # reindex item ids according to the igraph
        pos_iid += user_num

        user_path_sample_size = 0

        pos_iid_range = _check_temporal_causality_feasibility(temporal_matrix, pos_iid)
        if pos_iid_range is None:
            return set()

        # select new starting node
        pos_iid_idxs = np.arange(pos_iid_range)
        pos_iid_mask = np.ones(pos_iid_range, dtype=bool)
        while True:
            # select new starting node
            pos_iid_prob_mask = np.where(pos_iid_mask, pos_iid_mask / pos_iid_mask.sum(), 0)
            start_node_idx = np.random.choice(pos_iid_idxs, p=pos_iid_prob_mask)
            start_node = pos_iid[start_node_idx]
            pos_iid_mask[start_node_idx] = False

            if restrict_by_phase:
                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])
            else:
                item_candidates = np.concatenate([all_items[:start_node], all_items[start_node + 1 :]])

            # First hop is the relation user-item already addressed
            generated_paths = graph.get_all_simple_paths(
                start_node, to=item_candidates, cutoff=path_hop_length, mode="all"
            )
            random.shuffle(generated_paths)
            for path in generated_paths:
                full_path = (u, *path)
                valid_path = KnowledgePathDataset._check_kg_path(
                    full_path, user_num, item_num, check_last_node=True, collaborative_path=collaborative_path
                )

                if valid_path:
                    user_paths.add(full_path)
                    user_path_sample_size += 1

                if user_path_sample_size == max_paths_per_user:
                    break

            if user_path_sample_size == max_paths_per_user or not pos_iid_mask.any():
                break

        return user_paths

    return process_user


@numba.jit(nopython=True, parallel=True)
def _add_paths_relations_parallel(paths, paths_with_relations, relation_map):
    for path_idx in numba.prange(paths.shape[0]):
        path = paths[path_idx]
        for node_idx in np.arange(path.shape[0] - 1):
            start_path = node_idx * 2
            if path[node_idx] == -1 or path[node_idx + 1] == -1:
                break
            edge_id = relation_map[path[node_idx], path[node_idx + 1]]
            paths_with_relations[path_idx, start_path] = path[node_idx]
            paths_with_relations[path_idx, start_path + 1] = edge_id
            paths_with_relations[path_idx, start_path + 2] = path[node_idx + 1]
