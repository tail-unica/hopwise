import random
from itertools import chain, zip_longest
from string import Formatter

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers import models as token_models
from tokenizers import processors as token_processors
from tokenizers import trainers as token_trainers
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from hopwise.data import Interaction
from hopwise.data.dataset import KnowledgeBasedDataset
from hopwise.utils import PathLanuageModelingTokenType, set_color


class KnowledgePathDataset(KnowledgeBasedDataset):
    """:class:`KnowledgePathDataset` is based on :class:`~hopwise.data.dataset.KnowledgeBasedDataset`,
    and provides an interface to prepare tokenized knowledge graph path for path language modeling.

    Attributes:
        path_hop_length (int): The same as ``config["path_hop_length"]``.

        max_paths_per_user (int): The same as ``config["max_paths_per_user"]``.

        temporal_causality (bool): The same as ``config["temporal_causality"]``.

        collaborative_path (bool): The same as ``config["collaborative_path"]``.
        Not used when :attr:`path_sampling_strategy` = `metapaths` as
        collaborative metapaths must be explicitly defined.

        path_sampling_strategy (str): The same as ``config["path_sampling_strategy"]``.

        reasoning_path_template (str): The same as ``config["reasoning_path_template"]``.

        tokenizer (PreTrainedTokenizerFast): Tokenizer to process the sample paths.
    """

    PATH_PADDING = -1

    def __init__(self, config):
        super().__init__(config)
        self._path_dataset = None  # path dataset is generated with generate_path_dataset

        self._init_tokenizer()

    def _get_field_from_config(self):
        super()._get_field_from_config()

        # Path sampling parameters
        self.path_hop_length = self.config["path_hop_length"]
        self.max_paths_per_user = self.config["max_paths_per_user"]
        self.temporal_causality = self.config["temporal_causality"]
        self.collaborative_path = self.config["collaborative_path"]
        self.path_sampling_strategy = self.config["path_sampling_strategy"]
        self.reasoning_path_template = self.config["reasoning_path_template"]
        self.max_consecutive_invalid_paths_per_user = self.config["MAX_CONSECUTIVE_INVALID_PATHS_PER_USER"]

        # Tokenizer parameters
        self.tokenizer_model = self.config["tokenizer"]["model"]
        self.context_length = self.config["tokenizer"]["context_length"]
        self.sequence_template = self.config["tokenizer"]["template"]

        # Special tokens
        self.unk_token = self.config["tokenizer"]["special_tokens"]["unk_token"]
        self.pad_token = self.config["tokenizer"]["special_tokens"]["pad_token"]
        self.eos_token = self.config["tokenizer"]["special_tokens"]["eos_token"]
        self.bos_token = self.config["tokenizer"]["special_tokens"]["bos_token"]
        self.sep_token = self.config["tokenizer"]["special_tokens"]["sep_token"]
        self.cls_token = self.config["tokenizer"]["special_tokens"]["cls_token"]
        self.mask_token = self.config["tokenizer"]["special_tokens"]["mask_token"]
        self.special_tokens = list(self.config["tokenizer"]["special_tokens"].values())

        self.logger.debug(set_color("tokenizer", "blue") + f": {self.tokenizer_model}")

    @property
    def path_dataset(self):
        if self._path_dataset is None:
            self.logger.warning("Path dataset has not been generated yet, please call generate_path_dataset() first.")
        else:
            return self._path_dataset

    def _init_tokenizer(self):
        """Initialize tokenizer for the dataset."""
        tokenizer_model_class = getattr(token_models, self.tokenizer_model)

        tokenizer_object = Tokenizer(tokenizer_model_class(unk_token=self.unk_token))

        # Pre-tokenizer defined based on separator used between tokens in :attr:`reasoning_path_template`
        template_fields = list(Formatter().parse(self.reasoning_path_template))
        separator = template_fields[1][0]
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(separator, "removed")

        # The token vocabulary is generated and passed to the tokenizer trainer
        entity_range = np.arange(self.item_num + 1, self.entity_num)  # only entities that are not items are considered
        token_vocab = np.concatenate(
            [
                np.char.add(PathLanuageModelingTokenType.USER.value, np.arange(self.user_num).astype(str)),
                np.char.add(PathLanuageModelingTokenType.ITEM.value, np.arange(self.item_num).astype(str)),
                np.char.add(PathLanuageModelingTokenType.ENTITY.value, entity_range.astype(str)),
                np.char.add(PathLanuageModelingTokenType.RELATION.value, np.arange(self.relation_num).astype(str)),
            ]
        )

        tokenizer_trainer_class = getattr(token_trainers, self.tokenizer_model + "Trainer")
        tokenizer_trainer = tokenizer_trainer_class(
            vocab_size=len(token_vocab) + len(self.special_tokens), special_tokens=self.special_tokens
        )

        tokenizer_object.train_from_iterator(token_vocab, trainer=tokenizer_trainer)

        # Dyanmically formats the sequence template with the special tokens
        template_tokens = [parsed_string[1] for parsed_string in Formatter().parse(self.sequence_template)]
        try:
            template_token_map = {token: getattr(self, token) for token in template_tokens}
            sequence_template = self.sequence_template.format(**template_token_map)
        except AttributeError:
            raise AttributeError(
                f"The tokenizer sequence template with the field names [{template_tokens}] is not valid."
            )

        tokenizer_object.post_processor = token_processors.TemplateProcessing(
            single=sequence_template,
            special_tokens=[
                (spec_token, tokenizer_object.token_to_id(spec_token)) for spec_token in template_token_map.values()
            ],
        )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            model_max_length=self.context_length,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            mask_token=self.mask_token,
        )

    def tokenize_path_dataset(self, phase="train"):
        """Tokenize the path dataset.

        Args:
            phase (str, optional): The phase for which the path dataset is used. Defaults to "train".
        """

        def tokenization(example):
            return self.tokenizer(example["path"], truncation=True, padding=True, max_length=self.context_length)

        hf_path_dataset = HuggingFaceDataset.from_dict({"path": self.path_dataset.split("\n")})
        tokenized_dataset = hf_path_dataset.map(tokenization, batched=True, remove_columns=["path"])
        tokenized_dataset = DatasetDict({phase: tokenized_dataset})

        return tokenized_dataset

    def generate_path_dataset(self, used_ids, phase="train"):
        """Generate path dataset by sampling paths from the knowledge graph.

        Paths represent walks in the graph that connect :attr:`hop_length` + 1 entities through
        :attr:`hop_length` relations.
        Each path connects two items that a user interacted with.

        Refer to :meth:`generate_paths` for more details about path generation strategies.

        Args:
            used_ids (numpy.ndarray): The used ids.
        """
        if not isinstance(self.inter_feat, Interaction):
            raise ValueError("The data should be prepared before generating the path dataset.")

        if self.path_dataset is None:
            generated_paths = self.generate_paths(used_ids)

            path_string = ""
            for path in generated_paths:
                path_string += self._format_path(path) + "\n"

            self._path_dataset = path_string

        tokenized_dataset = self.tokenize_path_dataset(phase=phase)

        return tokenized_dataset

    def generate_paths(self, used_ids):
        """Generate paths from the knowledge graph.

        It currently supports four sampling strategies:

        - weighted-rw: sampling-and-discarding approach through weighted random walk.
                       Paths are sampled from the knowledge graph and discarded if they are not valid, i.e.,
                       they do not end in a positive item

        - constrained-rw: faithful random walk with constraints based on expected path output.

        - simple: randomly sample a positive item for each user and extract all simple paths to other positive items.

        - metapath: random walk constrained by pre-defined metapaths.

        Args:
            used_ids (numpy.ndarray): Positive item ids for each user.
            strategy (str, optional): The strategy for path generation. Defaults to "constrained-rw".

        Returns:
            list: List of paths with relations.
        """
        strategy = self.path_sampling_strategy
        if strategy in ["weighted-rw", "constrained-rw", "simple"]:
            graph = self._create_ckg_igraph(show_relation=True, directed=False)
        elif strategy in ["metapath"]:
            graph = self.ckg_hetero_graph(form="dgl", directed=not self.collaborative_path)
        else:
            raise NotImplementedError(f"Path generation method [{strategy}] has not been implemented.")

        temporal_matrix = None
        if self.temporal_causality:
            if self.time_field in self.inter_feat:
                user = self.inter_feat[self.uid_field].numpy()
                item = self.inter_feat[self.iid_field].numpy()
                timestamp = self.inter_feat[self.time_field].numpy()

                temporal_matrix = np.zeros((self.user_num, self.item_num), dtype=timestamp.dtype)
                temporal_matrix[user, item] = timestamp
            else:
                self.logger.warning(
                    "time_field has not been loaded or set,"
                    "thus temporal causality will not be used for path generation."
                )

        rw_args = self.config["random_walk_args"]
        strategy = strategy or self.path_sampling_strategy
        if strategy == "weighted-rw":
            if not self.collaborative_path:
                # prevent ui-relations to be sampled
                kg_rel_num = len(self.relations)
                graph.es["weight"] = [0.0] * (self.inter_num) + [1.0] * kg_rel_num
            else:
                graph.es["weight"] = [1.0] * graph.ecount()

            generated_paths = self._generate_paths_weighted_random_walk(
                graph,
                used_ids,
                temporal_matrix=temporal_matrix,
                max_tries_per_iid=rw_args["MAX_TRIES_PER_IID"],
                max_visited_duplicates_per_iid=rw_args["MAX_VISITED_DUPLICATES_PER_IID"],
            )
        elif strategy == "constrained-rw":
            generated_paths = self._generate_paths_constrained_random_walk(
                graph, used_ids, temporal_matrix=temporal_matrix, paths_per_hop=rw_args["MAX_PATHS_PER_HOP"]
            )
        elif strategy == "simple":
            generated_paths = self._generate_paths_all_simple(graph, used_ids, temporal_matrix=temporal_matrix)
        elif strategy == "metapath":
            generated_paths = self._generate_paths_from_metapaths(graph, used_ids, temporal_matrix=temporal_matrix)
        else:
            raise NotImplementedError(f"Path generation method [{strategy}] has not been implemented.")

        if strategy != "metapath":
            paths_with_relations = self._add_paths_relations(graph, generated_paths)
        else:
            padded_generated_paths = list(zip_longest(*generated_paths, fillvalue=self.PATH_PADDING))
            paths_with_relations = np.array(padded_generated_paths).T

        return paths_with_relations

    def _generate_paths_weighted_random_walk(
        self,
        graph,
        used_ids,
        temporal_matrix=None,
        max_tries_per_iid=50,
        max_visited_duplicates_per_iid=3,
    ):
        """Generate paths from the knowledge graph using weighted random walk.

        The last hop is not sampled, but it is selected according to the item candidates from temporal matrix
        if a relation between the second to last node and the item candidates exists.
        """
        paths = set()
        path_hop_length = self.path_hop_length - 2

        iter_users = tqdm(
            range(1, self.user_num),
            total=self.user_num - 1,
            ncols=100,
            desc=set_color("KG Path Sampling", "red"),
        )
        for u in iter_users:
            pos_iid = np.array(list(used_ids[u]))
            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            # reindex item ids according to the igraph
            pos_iid += self.user_num

            iid_tries = 0
            user_path_sample_size = 0
            user_invalid_paths = self.max_consecutive_invalid_paths_per_user
            while True:
                if iid_tries == 0:
                    iid_tries = max_tries_per_iid
                    duplicates = max_visited_duplicates_per_iid

                    # select new starting node
                    start_node_idx = np.random.randint(pos_iid.shape[0])
                    start_node = pos_iid[start_node_idx]

                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])

                # First hop is the relation user-item already addressed
                generated_path = graph.random_walk(start_node, path_hop_length, weights="weight")
                full_path = (u, *generated_path)

                valid_path = self._check_kg_path((*generated_path, -1), check_last_node=False)
                if valid_path:
                    reachable_candidates = graph.es.select(_source=full_path[-1], _target=item_candidates)
                    reachable_candidates = set(
                        e.source if e.source_vertex["type"] == self.iid_field else e.target
                        for e in reachable_candidates
                    )

                    if len(reachable_candidates) > 0:
                        last_node = np.random.choice(list(reachable_candidates))
                        full_path = (*full_path, last_node)
                    else:
                        valid_path = False

                if valid_path:
                    if full_path not in paths:
                        paths.add(full_path)
                        user_path_sample_size += 1
                        user_invalid_paths = self.max_consecutive_invalid_paths_per_user
                    else:
                        duplicates -= 1
                        user_invalid_paths -= 1
                else:
                    user_invalid_paths -= 1

                if duplicates == 0:
                    iid_tries = 0

                if user_path_sample_size == self.max_paths_per_user or user_invalid_paths == 0:
                    break

        return paths

    def _generate_paths_constrained_random_walk(self, graph, used_ids, temporal_matrix=None, paths_per_hop=1):
        """Generate paths from the knowledge graph using constrained random walks, similar to DGL random walk based on
        metapaths (https://docs.dgl.ai/en/1.1.x/generated/dgl.sampling.random_walk.html).

        The difference is that this strategy is constrained to the knowledge graph relations, but not to pre-defined
        metapahts. Then, the resulting paths may not be semantically sound, but they are still valid.

        Args:
            paths_per_hop (int, optional): The number of paths sampled at each hop to continue the random walk.
            Defaults to 1.
        """
        paths = set()
        path_hop_length = self.path_hop_length - 1

        iter_users = tqdm(
            range(1, self.user_num),
            total=self.user_num - 1,
            ncols=100,
            desc=set_color("KG Path Sampling", "red"),
        )
        for u in iter_users:
            pos_iid = np.array(list(used_ids[u]))
            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            # reindex item ids according to the igraph
            pos_iid += self.user_num

            user_path_sample_size = 0
            user_invalid_paths = self.max_consecutive_invalid_paths_per_user

            def _graph_traversal(g, path, hop, candidates):
                nonlocal paths
                nonlocal user_path_sample_size
                nonlocal user_invalid_paths

                if hop == 1:
                    next_node_candidates = g.es.select(_source=path[-1], _target=candidates)
                    next_node_candidates = list(
                        set(e.source if e.source_vertex != path[-1] else e.target for e in next_node_candidates)
                    )
                else:
                    next_node_candidates = []
                    for node in g.neighbors(path[-1]):
                        if self.collaborative_path:
                            type_check = g.vs[node]["type"] in [self.entity_field, self.uid_field]
                        else:
                            type_check = g.vs[node]["type"] == self.entity_field

                        # Self-loops are discarded
                        if node != path[-1] and type_check:
                            next_node_candidates.append(node)

                next_nodes = np.random.choice(
                    next_node_candidates, min(len(next_node_candidates), paths_per_hop), replace=False
                )
                for node in next_nodes:
                    new_path = (*path, node)
                    if hop == 1:
                        # Path is valid per construction
                        paths.add(new_path)
                        user_path_sample_size += 1
                    else:
                        _graph_traversal(g, new_path, hop - 1, candidates)

                    if user_path_sample_size == self.max_paths_per_user:
                        return

            while True:
                # select new starting node
                start_node_idx = np.random.randint(pos_iid.shape[0])
                start_node = pos_iid[start_node_idx]

                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])

                # First hop is the relation user-item already addressed
                curr_path_sample_size = user_path_sample_size
                _graph_traversal(graph, (u, start_node), path_hop_length, item_candidates)
                if user_path_sample_size - curr_path_sample_size == 0:
                    user_invalid_paths -= paths_per_hop
                else:
                    user_invalid_paths = self.max_consecutive_invalid_paths_per_user

                if user_path_sample_size == self.max_paths_per_user or user_invalid_paths <= 0:
                    break

        return paths

    def _generate_paths_all_simple(self, graph, used_ids, temporal_matrix=None):
        """Generate paths from the knowledge graph by extracting all simple paths for a randomly sampled item.
        Refer to igraph's https://python.igraph.org/en/stable/api/igraph.Graph.html#get_all_simple_paths.

        It considers all valid simple paths between each user and randomly sampled item,
        so resulting paths are not much diverse, as not all starting <user, item> pairs might be sampled.
        """
        paths = set()
        path_hop_length = self.path_hop_length - 1

        iter_users = tqdm(
            range(1, self.user_num),
            total=self.user_num - 1,
            ncols=100,
            desc=set_color("KG Path Sampling", "red"),
        )
        for u in iter_users:
            pos_iid = np.array(list(used_ids[u]))
            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            # reindex item ids according to the igraph
            pos_iid += self.user_num

            user_path_sample_size = 0
            pos_iid_idxs = np.arange(pos_iid.shape[0])
            pos_iid_mask = np.ones(pos_iid.shape[0], dtype=bool)
            while True:
                # select new starting node
                pos_iid_prob_mask = np.where(pos_iid_mask, pos_iid_mask / pos_iid_mask.sum(), 0)
                start_node_idx = np.random.choice(pos_iid_idxs, p=pos_iid_prob_mask)
                start_node = pos_iid[start_node_idx]
                pos_iid_mask[start_node_idx] = False

                if temporal_matrix is not None:
                    item_candidates = pos_iid[start_node_idx + 1 :]
                else:
                    item_candidates = np.concatenate([pos_iid[:start_node_idx], pos_iid[start_node_idx + 1 :]])

                # First hop is the relation user-item already addressed
                generated_paths = graph.get_all_simple_paths(
                    start_node, to=item_candidates, cutoff=path_hop_length, mode="all"
                )
                random.shuffle(generated_paths)
                for path in generated_paths:
                    full_path = (u, *path)
                    valid_path = self._check_kg_path(path, check_last_node=True)

                    if valid_path:
                        paths.add(full_path)
                        user_path_sample_size += 1

                    if user_path_sample_size == self.max_paths_per_user:
                        break

                if user_path_sample_size == self.max_paths_per_user or not pos_iid_mask.any():
                    break

        return paths

    def _generate_paths_from_metapaths(self, graph, used_ids, temporal_matrix=None):
        """Generate paths from pre-defined metapaths. Refer to DGL's random walk based on metapaths
        https://docs.dgl.ai/en/1.1.x/generated/dgl.sampling.random_walk.html for more details.
        """
        import dgl
        import torch

        final_paths = set()

        iter_users = tqdm(
            range(1, self.user_num),
            total=self.user_num - 1,
            ncols=100,
            desc=set_color("KG Path Sampling", "red"),
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
            user_invalid_paths = self.max_consecutive_invalid_paths_per_user
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

                        user_invalid_paths = self.max_consecutive_invalid_paths_per_user
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

    def _check_kg_path(self, path, check_last_node=False):
        """Check if the path is valid. The first node must be an item node and it assumes the user node is omitted.

        Args:
            path (list): The path to be checked.

            check_last_node (bool, optional): Whether to check the last node in the path.
            Defaults to ``False``.
        """
        path = np.array(path, dtype=int)
        graph_max_uid = self.user_num - 1
        graph_min_iid = 1 + self.user_num
        graph_max_iid = self.item_num - 1 + self.user_num

        first_node_check = graph_min_iid <= path[0] <= graph_max_iid
        if self.collaborative_path:
            valid_path = np.logical_or(path[1:-1] > graph_max_iid, path[1:-1] <= graph_max_uid).all()
        else:
            valid_path = (path[1:-1] > graph_max_iid).all()
        check_last_node = not check_last_node or graph_min_iid <= path[-1] <= graph_max_iid
        return first_node_check and valid_path and check_last_node

    def _format_path(self, path):
        """Format the path to a string according to :attr:`reasoning_path_template`.
        The template used for formatting expects the following fields:

        user: Starting user node of the path.
        pos_iid: Positive item that `user` interacted with.
        entity_list: List of entities in the path. It can include user and item entities.
        rec_iid: Item sampled as recommendation candidate.

        Args:
            path (list): The path to be formatted.
        """
        path_template = self.reasoning_path_template
        template_fields = list(Formatter().parse(path_template))
        separator = template_fields[1][0]

        path = path[path != self.PATH_PADDING]  # remove padding for shorter paths
        user = path[0]
        pos_iid = path[2] - self.user_num  # remap to dataset id
        rec_iid = path[-1] - self.user_num  # remap to dataset id
        path_entities = path[4:-1:2]
        path_relations = path[1::2]

        entity_mapped_list = []
        graph_min_iid = self.user_num
        graph_max_iid = self.item_num - 1 + self.user_num
        for e in path_entities:
            if graph_min_iid <= e <= graph_max_iid:
                entity_mapped_list.append(PathLanuageModelingTokenType.ITEM.value + str(e - self.user_num))
            elif e < graph_min_iid:
                entity_mapped_list.append(PathLanuageModelingTokenType.USER.value + str(e))
            else:
                entity_mapped_list.append(PathLanuageModelingTokenType.ENTITY.value + str(e - self.user_num))

        relation_mapped_list = [PathLanuageModelingTokenType.RELATION.value + str(r) for r in path_relations]

        path_string = path_template.format(
            user=PathLanuageModelingTokenType.USER.value + str(user),
            pos_iid=PathLanuageModelingTokenType.ITEM.value + str(pos_iid),
            entity_list=separator.join(entity_mapped_list),
            rec_iid=PathLanuageModelingTokenType.ITEM.value + str(rec_iid),
        )

        # removes repeated separators due to empty entity list
        path_string = path_string.replace(separator * 2, separator)

        interleaved_entities_relations = zip_longest(path_string.split(separator), relation_mapped_list)
        path_string = separator.join(list(chain(*interleaved_entities_relations))[:-1])

        return path_string

    def _add_paths_relations(self, graph, paths):
        complete_path_length = self.path_hop_length * 2 + 1
        paths_with_relations = np.full((len(paths), complete_path_length), fill_value=self.PATH_PADDING, dtype=int)
        for i, path in enumerate(paths):
            for node_idx in range(len(path) - 1):
                edge = graph.es.find(_source=path[node_idx], _target=path[node_idx + 1])
                edge_token = self.field2token_id[self.relation_field][edge["type"]]

                start_path, end_path = node_idx * 2, node_idx * 2 + 3
                paths_with_relations[i, start_path:end_path] = [path[node_idx], edge_token, path[node_idx + 1]]

        return paths_with_relations

    def __str__(self):
        info = [
            super().__str__(),
            f"The number of hops used for path sampling: {self.path_hop_length}",
            f"Maximum number of paths sampled per user: {self.max_paths_per_user}",
            f"The path sampling strategy: {self.path_sampling_strategy}",
            f"The tokenizer model: {self.tokenizer_model}",
        ]
        return "\n".join(info)
