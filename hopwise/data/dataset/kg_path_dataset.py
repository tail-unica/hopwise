# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

from dataclasses import dataclass
from itertools import chain, zip_longest

import numba
import numpy as np

from hopwise.data import Interaction
from hopwise.data.dataset import KnowledgeBasedDataset
from hopwise.utils import PathLanguageModelingTokenType, PathSamplingStrategy, progress_bar, set_color


@dataclass
class CSRGraph:
    """Container for CSR graph arrays used in parallel random walks.

    This dataclass bundles together the CSR sparse matrix components
    needed for efficient graph traversal in numba.

    Attributes:
        indptr: CSR row pointers (int64)
        indices: CSR column indices (int64)
        relations: Edge relation types (int64)
    """

    indptr: np.ndarray
    indices: np.ndarray
    relations: np.ndarray

    @classmethod
    def from_sparse_matrix(cls, csr_matrix):
        """Create CSRGraph from scipy sparse matrix with relation data.

        Args:
            csr_matrix: Scipy CSR matrix with relations stored in data field

        Returns:
            CSRGraph instance
        """
        indptr = csr_matrix.indptr.astype(np.int64)
        indices = csr_matrix.indices.astype(np.int64)
        relations = csr_matrix.data.astype(np.int64)

        return cls(indptr=indptr, indices=indices, relations=relations)

    def unpack(self):
        """Unpack arrays for passing to numba functions.

        Returns:
            tuple: (indptr, indices, relations)
        """
        return self.indptr, self.indices, self.relations


def _run_batched_numba(
    numba_func,
    batched_arrays,
    fixed_args_before=(),
    fixed_args_after=(),
    batch_size=50000,
    desc="Processing",
):
    """Run a numba function in batches with progress tracking.

    This helper enables progress monitoring for parallel numba operations by splitting
    the work into batches and updating a progress bar between batch executions.

    The progress bar shows total paths processed (jumping by batch_size each iteration)
    rather than number of batches, giving better visibility into actual progress.

    The numba function is called as:
        numba_func(*fixed_args_before, *batched_slices, *fixed_args_after)

    Args:
        numba_func: The numba-compiled function to call
        batched_arrays: List of arrays to slice per batch (must all have same length)
        fixed_args_before: Tuple of args to pass before batched arrays
        fixed_args_after: Tuple of args to pass after batched arrays
        batch_size: Number of samples per batch
        desc: Description for the progress bar

    Returns:
        Tuple of (paths, relations) concatenated from all batches
    """
    n_total = len(batched_arrays[0])
    if n_total == 0:
        return None, None

    n_batches = (n_total + batch_size - 1) // batch_size
    all_paths = []
    all_rels = []

    # Progress bar shows total paths, updated by batch_size each iteration
    pbar = progress_bar(
        total=n_total,
        ncols=100,
        desc=set_color(desc, "red", progress=True),
    )

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_total)
        actual_batch_size = batch_end - batch_start

        # Slice all batched arrays
        batch_slices = tuple(arr[batch_start:batch_end] for arr in batched_arrays)

        # Call numba function
        batch_paths, batch_rels = numba_func(*fixed_args_before, *batch_slices, *fixed_args_after)

        all_paths.append(batch_paths)
        all_rels.append(batch_rels)

        # Update progress bar by actual batch size (jumps)
        pbar.update(actual_batch_size)

    pbar.close()

    paths = np.concatenate(all_paths, axis=0)
    path_rels = np.concatenate(all_rels, axis=0)

    return paths, path_rels


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
    # Default batch size for progress tracking in numba-parallel operations
    # Smaller batches = more frequent progress updates but slightly more overhead
    PARALLEL_BATCH_SIZE = 50000

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
        self.strategy = PathSamplingStrategy(path_sample_args["strategy"])
        self.path_token_separator = path_sample_args["path_token_separator"]
        self.restrict_by_phase = path_sample_args["restrict_by_phase"]
        self.max_consecutive_invalid = path_sample_args["MAX_CONSECUTIVE_INVALID"]

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

            formatted_paths = [self._format_path(path) for path in generated_paths]
            path_string = "\n".join(formatted_paths)
            self._path_dataset = path_string

    def generate_user_paths(self):
        """Generate paths from the knowledge graph.

        It currently supports three sampling strategies:

        - weighted-rw: sampling-and-discarding approach through weighted random walk.
                       Paths are sampled from the knowledge graph and discarded if they are not valid, i.e.,
                       they do not end in a positive item

        - constrained-rw: faithful random walk with constraints based on expected path output.

        - simple-ui: extract all simple paths from users to all their positive items using BFS.

        - metapath: random walk constrained by pre-defined metapaths.

        Returns:
            list: List of paths with relations.
        """
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

        if self.strategy in [
            PathSamplingStrategy.WEIGHTED_RW,
            PathSamplingStrategy.CONSTRAINED_RW,
            PathSamplingStrategy.SIMPLE_UI,
        ]:
            # Build CSRGraph for efficient traversal
            csr_matrix = self._create_ckg_sparse_matrix(form="csr", show_relation=True)
            csr_graph = CSRGraph.from_sparse_matrix(csr_matrix)

            if self.strategy == PathSamplingStrategy.WEIGHTED_RW:
                max_tries_per_iid = self.config["path_sample_args"]["MAX_RW_TRIES_PER_IID"]
                paths_with_relations = self._generate_user_paths_weighted_random_walk(
                    csr_graph, used_ids, temporal_matrix=temporal_matrix, max_tries_per_iid=max_tries_per_iid
                )
            elif self.strategy == PathSamplingStrategy.CONSTRAINED_RW:
                max_paths_per_hop = self.config["path_sample_args"]["MAX_RW_PATHS_PER_HOP"]
                paths_with_relations = self._generate_user_paths_constrained_random_walk(
                    csr_graph, used_ids, temporal_matrix=temporal_matrix, paths_per_hop=max_paths_per_hop
                )
            elif self.strategy == PathSamplingStrategy.SIMPLE_UI:
                paths_with_relations = self._generate_user_paths_all_simple_ui(
                    csr_graph, used_ids, temporal_matrix=temporal_matrix
                )

        elif self.strategy == PathSamplingStrategy.METAPATH:
            graph = self.ckg_hetero_graph(form="dgl", directed=not self.collaborative_path)
            generated_paths = self._generate_user_paths_from_metapaths(
                graph, used_ids, temporal_matrix=temporal_matrix
            )
            padded_generated_paths = list(zip_longest(*generated_paths, fillvalue=self.PATH_PADDING))
            paths_with_relations = np.array(padded_generated_paths).T
        else:
            raise NotImplementedError(f"Path generation method [{self.strategy}] has not been implemented.")

        return paths_with_relations

    def _generate_user_paths_weighted_random_walk(
        self, csr_graph, used_ids, temporal_matrix=None, max_tries_per_iid=50
    ):
        """Generate paths from the knowledge graph using weighted random walk with CSR + numba.

        Uses batched parallel numba implementation for efficient path generation.

        Args:
            csr_graph: CSRGraph instance containing graph arrays
            used_ids: Array of sets containing positive item ids per user
            temporal_matrix: Optional temporal ordering matrix
            max_tries_per_iid: Maximum attempts per starting item
        """
        indptr, indices, relations = csr_graph.unpack()
        path_hop_length = self.path_hop_length - 2  # First and last hops handled separately
        graph_min_iid = np.int64(self.user_num)
        graph_max_iid = np.int64(self.item_num - 1 + self.user_num)

        # Prepare all start nodes and user mappings for batch processing
        all_start_nodes = []
        all_user_ids = []
        all_item_candidates = []  # For restrict_by_phase

        self.logger.info(set_color("Preparing batch data for weighted-rw...", "blue"))

        for u in range(1, self.user_num):
            pos_iid = np.array(list(used_ids[u]), dtype=np.int64)
            if len(pos_iid) == 0:
                continue

            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            pos_iid_graph = pos_iid + self.user_num

            if temporal_matrix is not None and len(pos_iid) <= 1:
                continue
            pos_iid_range = len(pos_iid) - (1 if temporal_matrix is not None else 0)

            # Generate multiple start nodes per user for batch processing
            n_samples_per_user = min(self.max_paths_per_user * max_tries_per_iid, pos_iid_range * max_tries_per_iid)
            start_indices = np.random.randint(0, pos_iid_range, size=n_samples_per_user)
            start_nodes = pos_iid_graph[start_indices]

            all_start_nodes.extend(start_nodes)
            all_user_ids.extend([u] * n_samples_per_user)

            # Store candidate info for each start
            for idx in start_indices:
                if self.restrict_by_phase:
                    if temporal_matrix is not None:
                        candidates = pos_iid_graph[idx + 1 :]
                    else:
                        candidates = np.concatenate([pos_iid_graph[:idx], pos_iid_graph[idx + 1 :]])
                    all_item_candidates.append(set(candidates))
                else:
                    all_item_candidates.append(None)

        if len(all_start_nodes) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        all_start_nodes = np.array(all_start_nodes, dtype=np.int64)
        all_user_ids = np.array(all_user_ids, dtype=np.int64)

        # Run random walks in batches with progress tracking
        paths, path_rels = _run_batched_numba(
            _csr_parallel_random_walks,
            batched_arrays=[all_start_nodes],
            fixed_args_before=(indptr, indices, relations),
            fixed_args_after=(path_hop_length, graph_min_iid, self.collaborative_path),
            batch_size=self.PARALLEL_BATCH_SIZE,
            desc="KG Path Sampling (weighted-rw)",
        )

        # Filter valid paths (no -1 in the middle)
        valid_mask = paths[:, -1] != -1

        # Check path structure: first node must be item, intermediate nodes valid
        first_node_valid = (paths[:, 0] >= graph_min_iid) & (paths[:, 0] <= graph_max_iid)
        valid_mask &= first_node_valid

        if not self.collaborative_path and path_hop_length > 1:
            # Intermediate nodes must not be users
            intermediate = paths[:, 1:-1]
            intermediate_valid = (intermediate >= graph_min_iid).all(axis=1) | (intermediate == -1).any(axis=1)
            valid_mask &= intermediate_valid

        valid_indices = np.where(valid_mask)[0]

        # Process valid paths and add final hop
        all_final_paths = []
        user_path_counts = {}
        ui_rel_id = self.field2token_id[self.relation_field][self.ui_relation]

        # Calculate target: max_paths_per_user * num_users for early stopping
        target_total_paths = self.max_paths_per_user * self.user_num

        pbar_validation = progress_bar(
            valid_indices,
            total=len(valid_indices),
            ncols=100,
            desc=set_color("Path Validation (weighted-rw)", "red", progress=True),
        )

        for idx in pbar_validation:
            u = all_user_ids[idx]

            # Check if user already has enough paths
            if user_path_counts.get(u, 0) >= self.max_paths_per_user:
                continue

            path = paths[idx]
            path_rel = path_rels[idx]
            start_node = all_start_nodes[idx]
            item_candidates = all_item_candidates[idx]

            # Find valid last hop to an item
            last_node = path[-1]
            neighbor_start = indptr[last_node]
            neighbor_end = indptr[last_node + 1]

            if neighbor_end <= neighbor_start:
                continue

            neighbor_nodes = indices[neighbor_start:neighbor_end]
            neighbor_rels = relations[neighbor_start:neighbor_end]

            # Filter to item nodes only
            item_mask = (neighbor_nodes >= graph_min_iid) & (neighbor_nodes <= graph_max_iid)
            item_mask &= neighbor_nodes != start_node

            if item_candidates is not None:
                item_mask &= np.array([n in item_candidates for n in neighbor_nodes])

            valid_neighbors = neighbor_nodes[item_mask]
            valid_rels = neighbor_rels[item_mask]

            if len(valid_neighbors) == 0:
                continue

            choice_idx = np.random.randint(len(valid_neighbors))
            final_node = valid_neighbors[choice_idx]
            final_rel = valid_rels[choice_idx]

            # Build interleaved path: user, rel, item, rel, ..., item
            path_with_rels = [u, ui_rel_id]

            for i in range(len(path)):
                path_with_rels.append(path[i])
                if i < len(path_rel):
                    path_with_rels.append(path_rel[i])

            path_with_rels.append(final_rel)
            path_with_rels.append(final_node)

            path_tuple = tuple(path_with_rels)
            all_final_paths.append(path_tuple)
            user_path_counts[u] = user_path_counts.get(u, 0) + 1

            # Early stopping: if we've collected enough paths, stop iterating
            if len(all_final_paths) >= target_total_paths:
                break

        # Deduplicate and convert to array
        unique_paths = list(set(all_final_paths))

        if len(unique_paths) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        complete_path_length = self.path_hop_length * 2 + 1
        paths_array = np.full((len(unique_paths), complete_path_length), fill_value=self.PATH_PADDING, dtype=np.int64)
        for i, path in enumerate(unique_paths):
            paths_array[i, : len(path)] = path

        return paths_array

    def _generate_user_paths_constrained_random_walk(self, csr_graph, used_ids, temporal_matrix=None, paths_per_hop=1):
        """Generate paths from the knowledge graph using constrained random walks with CSR + numba.

        Uses batched parallel numba implementation for efficient path generation.
        The walk is constrained to follow entity types (items -> entities -> items).

        Args:
            csr_graph: CSRGraph instance containing graph arrays
            used_ids: Array of sets containing positive item ids per user
            temporal_matrix: Optional temporal ordering matrix
            paths_per_hop: Number of paths to sample at each hop branching (used for oversampling)
        """
        indptr, indices, relations = csr_graph.unpack()
        path_hop_length = self.path_hop_length - 1  # First hop (user-item) handled separately
        graph_min_iid = np.int64(self.user_num)
        graph_max_iid = np.int64(self.item_num - 1 + self.user_num)

        # Prepare all start nodes and user mappings for batch processing
        all_start_nodes = []
        all_user_ids = []
        all_start_node_indices = []  # Index within user's pos_iid array

        # Build flattened positive item arrays for candidate lookup
        pos_iid_flat = []
        pos_iid_offsets = [0]  # pos_iid_offsets[u] gives start index in pos_iid_flat for user u

        self.logger.info(set_color("Preparing batch data for constrained-rw...", "blue"))

        for u in range(self.user_num):
            if u == 0:
                pos_iid_offsets.append(0)
                continue

            pos_iid = np.array(list(used_ids[u]), dtype=np.int64)
            if len(pos_iid) == 0:
                pos_iid_offsets.append(pos_iid_offsets[-1])
                continue

            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            pos_iid_graph = pos_iid + self.user_num
            pos_iid_flat.extend(pos_iid_graph)
            pos_iid_offsets.append(len(pos_iid_flat))

            if temporal_matrix is not None and len(pos_iid) <= 1:
                continue
            pos_iid_range = len(pos_iid) - (1 if temporal_matrix is not None else 0)

            # Generate multiple start nodes per user for batch processing
            # Oversample to account for invalid paths
            n_samples_per_user = self.max_paths_per_user * self.max_consecutive_invalid * paths_per_hop
            start_indices = np.random.randint(0, pos_iid_range, size=n_samples_per_user)
            start_nodes = pos_iid_graph[start_indices]

            all_start_nodes.extend(start_nodes)
            all_user_ids.extend([u] * n_samples_per_user)
            all_start_node_indices.extend(start_indices)

        if len(all_start_nodes) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        all_start_nodes = np.array(all_start_nodes, dtype=np.int64)
        all_user_ids = np.array(all_user_ids, dtype=np.int64)
        all_start_node_indices = np.array(all_start_node_indices, dtype=np.int64)
        pos_iid_flat = np.array(pos_iid_flat, dtype=np.int64)
        pos_iid_offsets = np.array(pos_iid_offsets, dtype=np.int64)

        # Run constrained random walks in batches with progress tracking
        paths, path_rels = _run_batched_numba(
            numba_func=_csr_constrained_random_walks,
            batched_arrays=[all_start_nodes],
            fixed_args_before=(indptr, indices, relations),
            fixed_args_after=(
                path_hop_length,
                graph_min_iid,
                graph_max_iid,
                self.collaborative_path,
            ),
            batch_size=self.PARALLEL_BATCH_SIZE,
            desc="KG Path Sampling (constrained-rw)",
        )

        # Filter valid paths (completed walks that end at items)
        valid_mask = paths[:, -1] != -1
        valid_mask &= (paths[:, -1] >= graph_min_iid) & (paths[:, -1] <= graph_max_iid)
        # Ensure end node is different from start node
        valid_mask &= paths[:, -1] != all_start_nodes

        # Apply restrict_by_phase filtering if needed
        if self.restrict_by_phase:
            phase_filter_indices = np.where(valid_mask)[0]
            pbar_phase = progress_bar(
                phase_filter_indices,
                total=len(phase_filter_indices),
                ncols=100,
                desc=set_color("Phase Filtering (constrained-rw)", "red", progress=True),
            )
            for idx in pbar_phase:
                u = all_user_ids[idx]
                start_idx = all_start_node_indices[idx]
                end_node = paths[idx, -1]

                # Get user's positive items
                user_pos_start = pos_iid_offsets[u]
                user_pos_end = pos_iid_offsets[u + 1]
                user_pos_items = pos_iid_flat[user_pos_start:user_pos_end]

                # Check if end_node is a valid candidate
                if temporal_matrix is not None:
                    # Only items after start_idx are valid
                    valid_candidates = user_pos_items[start_idx + 1 :]
                else:
                    # All items except the start item are valid
                    valid_candidates = (
                        np.concatenate([user_pos_items[:start_idx], user_pos_items[start_idx + 1 :]])
                        if len(user_pos_items) > 1
                        else np.array([], dtype=np.int64)
                    )

                if end_node not in valid_candidates:
                    valid_mask[idx] = False

        valid_indices = np.where(valid_mask)[0]

        # Build final paths with user and relations
        all_final_paths = []
        user_path_counts = {}
        ui_rel_id = self.field2token_id[self.relation_field][self.ui_relation]

        # Calculate target: max_paths_per_user * num_users for early stopping
        target_total_paths = self.max_paths_per_user * self.user_num

        pbar_validation = progress_bar(
            valid_indices,
            total=len(valid_indices),
            ncols=100,
            desc=set_color("Path Validation (constrained-rw)", "red", progress=True),
        )

        for idx in pbar_validation:
            u = all_user_ids[idx]

            # Check if user already has enough paths
            if user_path_counts.get(u, 0) >= self.max_paths_per_user:
                continue

            path = paths[idx]
            path_rel = path_rels[idx]

            # Build interleaved path: user, rel, item, rel, ..., item
            path_with_rels = [u, ui_rel_id]

            for i in range(len(path)):
                if path[i] == -1:
                    break
                path_with_rels.append(path[i])
                if i < len(path_rel) and path_rel[i] != -1:
                    path_with_rels.append(path_rel[i])

            path_tuple = tuple(path_with_rels)
            all_final_paths.append(path_tuple)
            user_path_counts[u] = user_path_counts.get(u, 0) + 1

            # Early stopping: if we've collected enough paths, stop iterating
            if len(all_final_paths) >= target_total_paths:
                break

        # Deduplicate and convert to array
        unique_paths = list(set(all_final_paths))

        if len(unique_paths) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        complete_path_length = self.path_hop_length * 2 + 1
        paths_array = np.full((len(unique_paths), complete_path_length), fill_value=self.PATH_PADDING, dtype=np.int64)
        for i, path in enumerate(unique_paths):
            paths_array[i, : len(path)] = path

        return paths_array

    def _generate_user_paths_all_simple_ui(self, csr_graph, used_ids, temporal_matrix=None):
        """Generate simple paths from users to their positive items using parallel random walks.

        This method uses parallel random walks with iterative re-sampling to find paths
        connecting positive items. It keeps retrying until all paths are found or
        no progress is made for max_consecutive_invalid consecutive attempts.

        Strategy:
        1. Run parallel random walks from all positive items needing paths
        2. Check coverage: count pairs that still need paths
        3. Re-sample: if missing pairs count unchanged for max_consecutive_invalid attempts, stop
        4. Otherwise, continue until all pairs are satisfied

        Args:
            csr_graph: CSRGraph instance containing graph arrays
            used_ids: Array of sets containing positive item ids per user
            temporal_matrix: Optional temporal ordering matrix
        """
        indptr, indices, relations = csr_graph.unpack()
        path_hop_length = self.path_hop_length - 1  # First hop (user-item) handled separately
        graph_min_iid = np.int64(self.user_num)
        graph_max_iid = np.int64(self.item_num - 1 + self.user_num)

        # Build user data structures
        user_pos_items = {}  # user -> list of positive item graph IDs
        user_pos_set = {}  # user -> set of positive item graph IDs (for fast lookup)

        self.logger.info(set_color("Preparing batch data for simple-ui...", "blue"))

        for u in range(1, self.user_num):
            pos_iid = np.array(list(used_ids[u]), dtype=np.int64)
            if len(pos_iid) == 0:
                continue

            if temporal_matrix is not None:
                pos_iid = pos_iid[np.argsort(temporal_matrix[u, pos_iid])]

            pos_iid_graph = pos_iid + self.user_num
            user_pos_items[u] = pos_iid_graph
            user_pos_set[u] = set(pos_iid_graph)

        if len(user_pos_items) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        # Paths per (user, positive_item) pair
        paths_per_pair = self.max_paths_per_user

        # Info and warning messages for simple-ui behavior
        self.logger.info(
            set_color(
                f"simple-ui: max_paths_per_user ({paths_per_pair}) limits paths per (user, positive_item) pair",
                "blue",
            )
        )

        # Warn if paths_per_pair seems high relative to average user actions
        if paths_per_pair > 0.1 * self.avg_actions_of_users:
            self.logger.warning(
                set_color(
                    f"max_paths_per_user ({paths_per_pair}) > 10% of avg_actions_of_users in training set "
                    f"({self.avg_actions_of_users:.2f}). This may result in excessive path generation.",
                    "yellow",
                )
            )

        # simple-ui forces restrict_by_phase by design (paths must connect positive items in order)
        if not self.restrict_by_phase:
            self.logger.info(set_color("simple-ui forces restrict_by_phase=True by design", "blue"))

        # Track paths per (user, start_item) pair
        # Key: (user_id, start_item_graph_id), Value: set of path tuples
        user_item_paths = {}
        ui_rel_id = self.field2token_id[self.relation_field][self.ui_relation]

        # Initialize tracking for all (user, start_item) pairs
        total_pairs = sum(len(pos_items) for pos_items in user_pos_items.values())
        for u, pos_items in user_pos_items.items():
            for start_item in pos_items:
                user_item_paths[(u, start_item)] = set()

        # Target: total paths we want to find
        target_total_paths = total_pairs * paths_per_pair

        # Tracking for early stopping
        consecutive_no_progress = 0
        prev_missing_pairs_count = total_pairs
        attempt = 0

        # Create a single progress bar showing total paths found
        pbar = progress_bar(
            total=target_total_paths,
            ncols=100,
            desc=set_color("KG Path Sampling (simple-ui)", "red", progress=True),
        )
        current_total_paths = 0

        # Iterative sampling until all paths found or no progress for max_consecutive_invalid attempts
        while True:
            attempt += 1

            # Find pairs that still need more paths
            pairs_needing_paths = [
                (u, start_item) for (u, start_item), paths in user_item_paths.items() if len(paths) < paths_per_pair
            ]

            missing_pairs_count = len(pairs_needing_paths)

            if missing_pairs_count == 0:
                self.logger.info(set_color(f"All pairs satisfied after {attempt} attempts", "green"))
                break

            # Check for progress: if missing pairs count unchanged, increment counter
            if missing_pairs_count == prev_missing_pairs_count:
                consecutive_no_progress += 1
                if consecutive_no_progress >= self.max_consecutive_invalid:
                    self.logger.info(
                        set_color(
                            f"No progress for {self.max_consecutive_invalid} consecutive attempts, "
                            f"stopping with {missing_pairs_count} pairs still missing paths",
                            "yellow",
                        )
                    )
                    break
            else:
                # Progress was made, reset counter
                consecutive_no_progress = 0
                prev_missing_pairs_count = missing_pairs_count

            # Prepare batch data for this iteration
            all_start_nodes = []
            all_user_ids = []

            # Oversample more aggressively to find paths faster
            samples_per_pair = max(4, paths_per_pair * 4)

            for u, start_item in pairs_needing_paths:
                needed = paths_per_pair - len(user_item_paths[(u, start_item)])
                n_samples = needed * samples_per_pair
                all_start_nodes.extend([start_item] * n_samples)
                all_user_ids.extend([u] * n_samples)

            if len(all_start_nodes) == 0:
                break

            all_start_nodes = np.array(all_start_nodes, dtype=np.int64)
            all_user_ids = np.array(all_user_ids, dtype=np.int64)

            # Run parallel random walks (without individual progress bar - we have the outer one)
            paths, path_rels = _csr_constrained_random_walks(
                indptr,
                indices,
                relations,
                all_start_nodes,
                path_hop_length,
                graph_min_iid,
                graph_max_iid,
                self.collaborative_path,
            )

            # Filter and validate paths
            valid_mask = paths[:, -1] != -1
            valid_mask &= (paths[:, -1] >= graph_min_iid) & (paths[:, -1] <= graph_max_iid)
            valid_mask &= paths[:, -1] != all_start_nodes  # End different from start

            # Single merged validation loop: check end node is positive item AND respects temporal order
            # simple-ui always enforces restrict_by_phase behavior
            for idx in np.where(valid_mask)[0]:
                u = all_user_ids[idx]
                start_node = all_start_nodes[idx]
                end_node = paths[idx, -1]

                # End node must be in user's positive items
                if end_node not in user_pos_set[u]:
                    valid_mask[idx] = False
                    continue

                # Enforce restrict_by_phase: end must be valid relative to start
                pos_items = user_pos_items[u]
                start_idx = np.where(pos_items == start_node)[0]
                if len(start_idx) == 0:
                    valid_mask[idx] = False
                    continue
                start_idx = start_idx[0]

                if temporal_matrix is not None:
                    # End must come after start in temporal order
                    valid_candidates = set(pos_items[start_idx + 1 :])
                else:
                    # Any other positive item is valid
                    valid_candidates = set(pos_items) - {start_node}

                if end_node not in valid_candidates:
                    valid_mask[idx] = False

            valid_indices = np.where(valid_mask)[0]

            # Process valid paths
            new_paths_found = 0
            for idx in valid_indices:
                u = all_user_ids[idx]
                start_node = all_start_nodes[idx]
                key = (u, start_node)

                # Check if this pair already has enough paths
                if len(user_item_paths[key]) >= self.max_paths_per_user:
                    continue

                path = paths[idx]
                path_rel = path_rels[idx]

                # Build interleaved path: user, ui_rel, item, rel, ..., item
                path_with_rels = [u, ui_rel_id]

                for i in range(len(path)):
                    if path[i] == -1:
                        break
                    path_with_rels.append(path[i])
                    if i < len(path_rel) and path_rel[i] != -1:
                        path_with_rels.append(path_rel[i])

                path_tuple = tuple(path_with_rels)
                if path_tuple not in user_item_paths[key]:
                    user_item_paths[key].add(path_tuple)
                    new_paths_found += 1

            # Update progress bar with new paths found
            if new_paths_found > 0:
                current_total_paths += new_paths_found
                pbar.update(new_paths_found)
                # Update description with attempt info
                pbar.set_description(
                    set_color(f"KG Path Sampling (simple-ui, attempt {attempt})", "red", progress=True)
                )

        pbar.close()

        # Collect all paths - each pair is already limited to max_paths_per_user during sampling
        all_final_paths = [path_tuple for paths_set in user_item_paths.values() for path_tuple in paths_set]

        if len(all_final_paths) == 0:
            return np.array([], dtype=np.int64).reshape(0, self.path_hop_length * 2 + 1)

        complete_path_length = self.path_hop_length * 2 + 1
        paths_array = np.full(
            (len(all_final_paths), complete_path_length), fill_value=self.PATH_PADDING, dtype=np.int64
        )
        for i, path in enumerate(all_final_paths):
            paths_array[i, : len(path)] = path

        return paths_array

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

    def __str__(self):
        info = [
            super().__str__(),
            f"The number of hops used for path sampling: {self.path_hop_length}",
            f"Maximum number of paths sampled per user: {self.max_paths_per_user}",
            f"The path sampling strategy: {self.strategy}",
            f"The tokenizer model: {self.tokenizer_model}",
        ]
        return "\n".join(info)


# ============================================================================
# Numba-accelerated helper functions for CSR-based random walks
# ============================================================================


@numba.njit(parallel=True)
def _csr_parallel_random_walks(indptr, indices, relations, start_nodes, num_steps, graph_min_iid, collaborative_path):
    """Parallel random walks on CSR graph.

    Note: Set np.random.seed() before calling this function for reproducibility.

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        relations: Edge relation types
        start_nodes: Array of starting nodes
        num_steps: Number of steps per walk
        graph_min_iid: Minimum item ID in graph (users have id < graph_min_iid)
        collaborative_path: If True, allow user nodes as intermediate nodes

    Returns:
        paths: (n_walks, num_steps + 1) node paths
        path_relations: (n_walks, num_steps) relation paths
    """
    n_walks = len(start_nodes)
    paths = np.full((n_walks, num_steps + 1), -1, dtype=np.int64)
    path_relations = np.full((n_walks, num_steps), -1, dtype=np.int64)

    for i in numba.prange(n_walks):
        node = start_nodes[i]
        paths[i, 0] = node

        for step in range(num_steps):
            start_idx = indptr[node]
            end_idx = indptr[node + 1]
            n_neighbors = end_idx - start_idx

            if n_neighbors == 0:
                break

            # Build list of valid neighbors
            valid_count = 0
            valid_indices = np.empty(n_neighbors, dtype=np.int64)

            for j in range(n_neighbors):
                neighbor = indices[start_idx + j]

                # If not collaborative_path, skip user nodes (id < graph_min_iid)
                if not collaborative_path and neighbor < graph_min_iid:
                    continue

                valid_indices[valid_count] = j
                valid_count += 1

            if valid_count == 0:
                break

            # Uniform random selection from valid neighbors
            selected = np.random.randint(valid_count)
            selected_idx = valid_indices[selected]

            neighbor_idx = start_idx + selected_idx
            node = indices[neighbor_idx]
            paths[i, step + 1] = node
            path_relations[i, step] = relations[neighbor_idx]

    return paths, path_relations


@numba.njit(parallel=True)
def _csr_constrained_random_walks(
    indptr,
    indices,
    relations,
    start_nodes,
    num_steps,
    graph_min_iid,
    graph_max_iid,
    collaborative_path,
):
    """Parallel constrained random walks on CSR graph.

    Constraints:
    - Start from item
    - End at item (different from start)
    - If collaborative_path=True: user nodes CAN be intermediate nodes
    - If collaborative_path=False: user nodes are NOT allowed as intermediate nodes

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        relations: Edge relation types
        start_nodes: Array of starting nodes (items)
        num_steps: Number of steps per walk
        graph_min_iid: Minimum item ID in graph (users have id < graph_min_iid)
        graph_max_iid: Maximum item ID in graph
        collaborative_path: If True, allow user nodes as intermediate nodes

    Returns:
        paths: (n_walks, num_steps + 1) node paths
        path_relations: (n_walks, num_steps) relation paths
    """
    n_walks = len(start_nodes)
    paths = np.full((n_walks, num_steps + 1), -1, dtype=np.int64)
    path_relations = np.full((n_walks, num_steps), -1, dtype=np.int64)

    for i in numba.prange(n_walks):
        start_node = start_nodes[i]
        node = start_node
        paths[i, 0] = node

        # Track visited nodes to avoid cycles (max path length is small)
        visited = np.zeros(num_steps + 1, dtype=np.int64)
        visited[0] = node
        n_visited = 1

        for step in range(num_steps):
            start_idx = indptr[node]
            end_idx = indptr[node + 1]
            n_neighbors = end_idx - start_idx

            if n_neighbors == 0:
                break

            is_last_step = step == num_steps - 1

            # Build list of valid neighbors based on constraints
            valid_count = 0
            valid_indices = np.empty(n_neighbors, dtype=np.int64)

            for j in range(n_neighbors):
                neighbor = indices[start_idx + j]

                # Check if already visited
                is_visited = False
                for v in range(n_visited):
                    if visited[v] == neighbor:
                        is_visited = True
                        break
                if is_visited:
                    continue

                if is_last_step:
                    # Last step: must end at item (not the start node)
                    if neighbor >= graph_min_iid and neighbor <= graph_max_iid:
                        if neighbor != start_node:
                            valid_indices[valid_count] = j
                            valid_count += 1
                # Intermediate step
                elif collaborative_path:
                    # Allow any node (users, items, entities)
                    valid_indices[valid_count] = j
                    valid_count += 1
                # Only allow items and entities (no users)
                elif neighbor >= graph_min_iid:
                    valid_indices[valid_count] = j
                    valid_count += 1

            if valid_count == 0:
                break

            # Uniform random selection from valid neighbors
            selected = np.random.randint(valid_count)
            selected_idx = valid_indices[selected]

            neighbor_idx = start_idx + selected_idx
            node = indices[neighbor_idx]
            paths[i, step + 1] = node
            path_relations[i, step] = relations[neighbor_idx]

            visited[n_visited] = node
            n_visited += 1

    return paths, path_relations
