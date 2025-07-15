# @Time   : 2025/02/19
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""CAFE
##################################################
Reference:
    Xian et al. "CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation." in CIKM 2020.

Reference code:
    https://github.com/orcax/CAFE
"""

import random
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hopwise.model.abstract_recommender import KnowledgeRecommender
from hopwise.utils import InputType


class CAFE(KnowledgeRecommender):
    """
    CAFE is a knowledge-aware recommender system that uses symbolic reasoning
    over a knowledge graph to explain recommendations.

    Note:
        Assumes that each relation corresponds to a unique pair of entity types. e.g. ui-relation -> (user, item)
    """

    input_type = InputType.USERWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.dataset = dataset

        # Load parameters info from config
        self.device = config["device"]
        self.load_embeddings = config["load_embeddings"]
        self.raw_metapaths = config["path_constraint"]

        # Load CAFE parameters
        self.rank_weight = config["rank_weight"]
        self.deep_module = config["deep_module"]
        self.use_dropout = config["use_dropout"]
        self.topk_candidates = config["topk_candidates"]
        self.sample_size = config["sample_size"]
        self.topk_paths = config["topk_paths"]
        self.path_max_user_trials = config["max_user_trials"]

        # user-item relation
        self.ui_relation = dataset.ui_relation
        self.ui_relation_id = dataset.field2token_id["relation_id"][dataset.ui_relation]

        # Load Knowledge Graph Embedding Checkpoint
        self.user_embedding = dataset.get_preload_weight("user_embedding_id")
        self.entity_embedding = dataset.get_preload_weight("entity_embedding_id")
        self.relation_embedding = dataset.get_preload_weight("relation_embedding_id")

        # Topk Candidates
        self.topk_user_items = self._compute_top_items()
        # Turn into torch, so that the weight is updated.
        self.user_embedding = torch.from_numpy(self.user_embedding).to(device=self.device, dtype=torch.float32)
        self.entity_embedding = torch.from_numpy(self.entity_embedding).to(self.device, dtype=torch.float32)
        self.relation_embedding = torch.from_numpy(self.relation_embedding).to(self.device, dtype=torch.float32)
        self.embedding_size = self.user_embedding.size(1)

        # Embedding mapping
        self.embeddings = {
            "user": self.user_embedding,
            "entity": self.entity_embedding,
            "relation": self.relation_embedding,
        }

        # Map Relation ID to relation name to check has pattern constraint
        self.rid2relation = {v: k for k, v in dataset.field2token_id["relation_id"].items()}
        self.relation2rid = dataset.field2token_id["relation_id"]

        # Positives
        self.positives = dataset.history_item_matrix()[0]

        # Load Full Knowledge Graph in dict form
        self.graph_dict = dataset.ckg_dict_graph(ui_bidirectional=False)
        self.memory_size = 10000  # number of paths to save for each metapath
        self.replay_memory = {}

        self.relation_info = dict()
        for relation_name in self.rid2relation.values():
            if relation_name == "[PAD]":
                continue

            if relation_name == f"{dataset.ui_relation}_r":
                raise ValueError("The ui_relation name should not end with '_r'.")

            if relation_name == dataset.ui_relation:
                head, tail = "user", "entity"
            else:
                head, tail = "entity", "entity"

            if relation_name not in self.relation_info:
                self.relation_info[relation_name] = {"name": relation_name, "entity_head": head, "entity_tail": tail}

        # Transform each node in the metapath in a tuple
        self.metapaths = []
        for mp in self.raw_metapaths:
            metapath = []
            for node in mp:
                metapath.append((node[0], node[1]))
            self.metapaths.append(metapath)

        self.mpath_ids = list(range(len(self.metapaths)))

        for mpid in range(len(self.metapaths)):
            self.replay_memory[mpid] = ReplayMemory(self.memory_size)

        self.model = SymbolicNetwork(
            self.relation_info,
            self.relation2rid,
            self.embeddings,
            self.embedding_size,
            self.deep_module,
            self.use_dropout,
            self.n_items,
            self.device,
        )

        # random generator
        self.rng = np.random.default_rng()

    def _compute_top_items(self):
        u_p_scores = np.dot(
            self.user_embedding + self.relation_embedding[self.ui_relation_id], self.entity_embedding[: self.n_items].T
        )
        ui_scores = np.argsort(u_p_scores, axis=1)  # From worst to best
        top100_ui_scores = ui_scores[:, -100:][:, ::-1]
        topk_user_items = top100_ui_scores[:, : self.topk_candidates]
        return topk_user_items

    def _get_batch_by_user(self, users):
        pos_path_batch, neg_pid_batch = [], []
        top_pids = np.arange(self.topk_candidates)
        user_trials = {u.item(): 0 for u in users}  # Track trials per user
        skipped_users = []
        # it select the it user, if a path is not found, another metapath and item is used.
        it = 0
        while len(pos_path_batch) < len(users) and it < len(users):
            # Take the user
            user = users[it].item()

            # Skip if max trials exceeded
            if user_trials[user] >= self.path_max_user_trials:
                if user not in skipped_users:
                    skipped_users.append(user)
                it += 1
                continue
            # Sample a metapath
            mpid = self.rng.choice(self.mpath_ids)

            # Sample one of the best topk_candidates e.g. sample one of the top20 pids
            pidx = self.rng.choice(top_pids)

            # Take the corresponding score
            item = self.topk_user_items[user][pidx]

            # Compute the probability to sample path from memory, P \in [0, 0.5].
            use_memory_prob = 0.5 * len(self.replay_memory[mpid]) / self.memory_size

            # Sample a history path from memory.
            if self.rng.random() < use_memory_prob:
                hist_path = self.replay_memory[mpid].sample()
                pos_path_batch.append(hist_path)
            # Sample a new path from graph.
            else:
                paths = self.fast_sample_path_with_target(mpid, user, item, 1)
                # if a path is not found, try again
                if not paths:
                    user_trials[user] += 1
                    continue
                pos_path_batch.append(paths[0])
                self.replay_memory[mpid].add(paths)

            # Sample a negative item.
            if pidx < self.topk_candidates - 1:
                neg_pidx = self.rng.choice(np.arange(pidx + 1, self.topk_candidates))
                neg_pid = self.topk_user_items[user][neg_pidx]
            else:
                neg_pid = self.rng.choice(self.n_items)

            neg_pid_batch.append(neg_pid)
            it += 1

        pos_path_batch = np.array(pos_path_batch)
        neg_pid_batch = np.array(neg_pid_batch)
        return mpid, pos_path_batch, neg_pid_batch

    def _rev_rel(self, rel):
        if rel == self.ui_relation:
            return self.ui_relation

        if rel.endswith("_r"):
            return rel[:-2]
        return rel + "_r"

    def fast_sample_path_with_target(self, mpath_id, user_id, target_id, num_paths, sample_size=100):
        """Sample one path given source and target, using BFS from both sides.
        Returns:
            list of entity ids.
        """
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_paths = [[user_id]]
        for i in range(1, mid_level + 1):
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_paths = []
            for fp in forward_paths:
                try:
                    next_ids = self.graph_dict[last_entity][fp[-1]][self.relation2rid[next_relation]]
                except KeyError:
                    next_ids = []
                # Random sample ids
                if len(next_ids) > sample_size:
                    # next_ids = np.random.permutation(next_ids)[:sample_size]
                    next_ids = self.rng.choice(next_ids, size=sample_size, replace=False)
                for next_id in next_ids:
                    tmp_paths.append(fp + [next_id])
            forward_paths = tmp_paths
        # Backward BFS (e.g. e4--e5--e6).
        backward_paths = [[target_id]]
        for i in reversed(range(mid_level + 2, path_len + 1)):  # i=l, l-2,..., mid+2
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for bp in backward_paths:
                try:
                    curr_ids = self.graph_dict[next_entity][bp[0]][self.relation2rid[self._rev_rel(next_relation)]]
                except KeyError:
                    curr_ids = []
                # Random sample ids
                if len(curr_ids) > sample_size:
                    curr_ids = self.rng.choice(curr_ids, size=sample_size, replace=False)
                for curr_id in curr_ids:
                    tmp_paths.append([curr_id] + bp)
            backward_paths = tmp_paths
        # Build hash map for indexing backward paths.
        # e.g. a dict with key=e3 and value=(e4--e5--e6).
        backward_map = {}
        next_relation, next_entity = metapath[mid_level + 1]
        # convert relation name to id for graphdict lookup
        for bp in backward_paths:
            try:
                curr_ids = self.graph_dict[next_entity][bp[0]][self.relation2rid[self._rev_rel(next_relation)]]
            except KeyError:
                try:
                    relations = list(self.graph_dict[last_entity][fp[-1]].keys())
                    curr_ids = self.graph_dict[last_entity][fp[-1]][self.rng.choice(relations)]
                except KeyError:
                    curr_ids = []
            if len(curr_ids) > sample_size:
                curr_ids = self.rng.choice(curr_ids, size=sample_size, replace=False)
            for curr_id in curr_ids:
                if curr_id not in backward_map:
                    backward_map[curr_id] = []
                backward_map[curr_id].append(bp)
        # Find intersection of forward paths and backward paths.
        final_paths = []
        for fp_idx in self.rng.permutation(len(forward_paths)):
            fp = forward_paths[fp_idx]
            mid_id = fp[-1]
            if mid_id not in backward_map:
                continue
            self.rng.shuffle(backward_map[mid_id])
            for bp in backward_map[mid_id]:
                final_paths.append(fp + bp)
                if len(final_paths) >= num_paths:
                    break
            if len(final_paths) >= num_paths:
                break

        return final_paths

    def count_paths_with_target(self, mpath_id, user_id, target_id, sample_size=50):
        """This is an approx count, not exact."""
        if isinstance(target_id, torch.Tensor):
            target_id = target_id.item()
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_ids = [user_id]
        for i in range(1, mid_level + 1):  # i=1, 2,..., mid
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_ids = []
            for eid in forward_ids:
                try:
                    next_ids = self.graph_dict[last_entity][eid][self.relation2rid[next_relation]]
                    if len(next_ids) > sample_size:
                        next_ids = self.rng.choice(next_ids, size=sample_size, replace=False).tolist()
                    tmp_ids.extend(next_ids)
                except KeyError:
                    continue
            forward_ids = tmp_ids

        # Backward BFS (e.g. e4--e5--e6).
        backward_ids = [target_id]
        for i in reversed(range(mid_level + 1, path_len + 1)):  # i=l, l-1,..., mid+1
            next_relation, next_entity = metapath[i]
            tmp_ids = []
            for eid in backward_ids:
                try:
                    curr_ids = self.graph_dict[next_entity][eid][self.relation2rid[self._rev_rel(next_relation)]]
                except KeyError:
                    curr_ids = []
                tmp_ids.extend(curr_ids)
            backward_ids = tmp_ids

        count = len(set(forward_ids).intersection(backward_ids))
        return count

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        users = users[users != 0]

        mpid, pos_paths, neg_pids = self._get_batch_by_user(users)

        pos_paths = torch.from_numpy(pos_paths).to(self.device)
        neg_pids = torch.from_numpy(neg_pids).to(self.device)
        reg_loss, rank_loss = self.model.forward(self.metapaths[mpid], pos_paths, neg_pids)
        rank_loss *= self.rank_weight

        return reg_loss, rank_loss

    def predict(self, interaction):
        return

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        kg_mask = KGMask(self.graph_dict, self.ui_relation_id)
        predicted_paths = self._infer_paths(users, kg_mask)
        path_counts = self._estimate_path_count(users)
        results = self.run_program(users, path_counts, predicted_paths)
        scores, paths = results
        paths = self.convert_path_relations(paths)
        return scores, paths

    def convert_path_relations(self, paths):
        new_data = []
        for user, item, score, path in paths:
            sanitized_path = [path[0]] + [
                (self.relation2rid[relation], e_type, eid) for relation, e_type, eid in path[1:]
            ]
            new_data.append([user, item, score, sanitized_path])
        return new_data

    def explain(self, interaction):
        """Support function used for case study.

        Args:
            interaction : test interaction data

        Returns:
            pd.Dataframe: explanation results with columns: "user", "item", "score", "path"
        """
        users = interaction[self.USER_ID]

        kg_mask = KGMask(self.graph_dict, self.ui_relation_id)
        predicted_paths = self._infer_paths(users, kg_mask)
        path_counts = self._estimate_path_count(users)

        scores, explanations = self.run_program(users, path_counts, predicted_paths)

        for exp in explanations:
            exp[-1] = self.decode_path(exp[-1])

        return scores, explanations

    def decode_path(self, path):
        return path

    def _infer_paths(self, users, kg_mask):
        predictions = dict()
        for user in users:
            user = user.item()  # noqa: PLW2901

            predictions[user] = dict()
            for mpid, metapath in enumerate(self.metapaths):
                paths = self.model.infer_with_path(
                    metapath, user, kg_mask, excluded_pids=self.positives[user], topk_paths=self.topk_paths
                )
                predictions[user][mpid] = paths
        return predictions

    def _estimate_path_count(self, users):
        num_mp = len(self.metapaths)
        counts = dict()
        for user in users:
            user = user.item()  # noqa: PLW2901

            counts[user] = np.zeros(num_mp)
            positives = self.positives[user]
            positives = positives[positives != 0]  # due to padding
            for pid in positives:
                for mpid in range(num_mp):
                    cnt = self.count_paths_with_target(mpid, user, pid, 50)
                    counts[user][mpid] += cnt
            counts[user] = counts[user] / len(self.positives[user])
        return counts

    def run_program(self, users, path_counts, predicted_paths):
        results = torch.full((len(users), self.n_items), -torch.inf)
        collect_results = list()

        kg_mask = KGMask(self.graph_dict, self.ui_relation_id)
        program_exe = MetaProgramExecutor(self.model, self.rng, self.device, kg_mask, self.relation2rid)

        pred_paths_instances = dict()
        for i, user in enumerate(users):
            user = user.item()  # noqa: PLW2901

            pred_paths_instances[user] = dict()
            program = self.create_heuristic_program(self.metapaths, predicted_paths[user], path_counts[user])
            positives = self.positives[user]
            positives = positives[positives != 0]  # due to padding
            program_exe.execute(program, user, positives)
            paths = program_exe.collect_results(program)
            tmp = [(r[0][-1], reduce(lambda x, y: x * y, r[1])) for r in paths]
            for r in paths:
                path = [("self_loop", "user", r[0][0])]
                for j in range(len(r[-1])):
                    path.append((r[-1][j], r[2][j], r[0][j + 1]))
                    # stop when a path is created
                    if j == len(r[-1]) - 1:
                        continue
                pred_paths_instances[r[0][0]][r[0][-1]] = (reduce(lambda x, y: x * y, r[1]), np.mean(r[1][-1]), path)

            top_items_scores = sorted(tmp, key=lambda x: x[1], reverse=True)
            for item, score in top_items_scores:
                if item < self.n_items and results[i, item] < score:  # if it's an item
                    results[i, item] = score.tolist()
                    collect_results.append([user, item, score, pred_paths_instances[user][item][2]])

        return results, collect_results

    def create_heuristic_program(self, metapaths, predicted_paths, path_counts):
        pcount = path_counts.astype(np.float32)
        pcount[pcount > 5] = 5  # noqa: PLR2004

        mp_scores = np.ones(len(metapaths)) * -99
        for mpid in predicted_paths:
            paths = predicted_paths[mpid]
            if len(paths) <= 0:
                continue
            scores = np.array([p2[-1] for _, p2 in paths])
            scores[scores < -5.0] = -5.0  # noqa: PLR2004
            mp_scores[mpid] = np.mean(scores)
        top_idxs = np.argsort(mp_scores)[::-1]

        norm_count = np.zeros(len(metapaths))
        rest = self.sample_size
        for mpid in top_idxs:
            if pcount[mpid] <= rest:
                norm_count[mpid] = pcount[mpid]
            else:
                norm_count[mpid] = rest
            rest -= norm_count[mpid]

        program_layout = NeuralProgramLayout(metapaths)
        program_layout.update_by_path_count(norm_count)

        return program_layout


class SymbolicNetwork(nn.Module):
    def __init__(
        self, relation_info, relation2rid, embeddings, embedding_size, deep_module, use_dropout, n_items, device
    ):
        super().__init__()
        self.embedding = embeddings
        self.embedding_size = embedding_size
        self.n_items = n_items
        self.device = device
        self.relation2rid = relation2rid

        self._create_modules(relation_info, deep_module, use_dropout)
        self.ce_loss = nn.CrossEntropyLoss()

    def _create_modules(self, relation_info, use_deep=False, use_dropout=True):
        """Create module for each relation."""
        for name in relation_info:
            info = relation_info[name]
            if not use_deep:
                module = RelationModule(self.embedding_size, info)
            else:
                module = DeepRelationModule(self.embedding_size, info, use_dropout)
            setattr(self, name, module)

    def _get_modules(self, metapath):
        """Get list of modules by metapath."""
        module_seq = []  # seq len = len(metapath)-1
        for relation, _ in metapath[1:]:
            module = getattr(self, relation)
            module_seq.append(module)
        return module_seq

    def _forward(self, modules, uids):
        outputs = []
        batch_size = uids.size(0)

        user_vec = self.embedding["user"][uids]  # [bs, d]
        input_vec = user_vec
        for module in modules:
            out = module((input_vec, user_vec)).view(batch_size, -1)  # [bs, d]
            outputs.append(out)
            input_vec = out
        return outputs

    def forward(self, metapath, pos_paths, neg_pids):
        """Compute loss.
        Args:
            metapath: list of relations, e.g. [USER, (r1, e1),..., (r_n, e_n)].
            uid: a LongTensor of user ids, with size [bs, ].
            target_path: a LongTensor of node ids, with size [bs, len(metapath)],
                    e.g. each path contains [u, e1,..., e_n].
        Returns:
            logprobs: sum of log probabilities of given target node ids, with size [bs, ].
        """
        modules = self._get_modules(metapath)
        outputs = self._forward(modules, pos_paths[:, 0])

        # Path regularization loss
        reg_loss = 0
        scores = 0
        for i, module in enumerate(modules):
            et_vecs = self.embedding[module.et_name]
            scores = torch.matmul(outputs[i], et_vecs.t())
            reg_loss += self.ce_loss(scores, pos_paths[:, i + 1])

        # Ranking loss
        logprobs = F.log_softmax(scores, dim=1)  # [bs, vocab_size]
        pos_score = torch.gather(logprobs, 1, pos_paths[:, -1].view(-1, 1))
        neg_score = torch.gather(logprobs, 1, neg_pids.view(-1, 1))
        rank_loss = torch.sigmoid(neg_score - pos_score).mean()

        return reg_loss, rank_loss

    def forward_simple(self, metapath, uids, pids):
        modules = self._get_modules(metapath)
        outputs = self._forward(modules, uids)

        # Path regularization loss
        items = self.embedding["entity"].weight[: self.n_items]  # [bs, d]
        scores = torch.matmul(outputs[-1], items.t())  # [bs, vocab_size]
        logprobs = F.log_softmax(scores, dim=1)  # [bs, vocab_size]
        pid_logprobs = logprobs.gather(1, pids.view(-1, 1)).view(-1)
        return pid_logprobs

    def infer_direct(self, metapath, uid, pids):
        if len(pids) == 0:
            return []
        modules = self._get_modules(metapath)
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        # list of tensor of [1, d]
        outputs = self._forward(modules, uid_tensor)

        # Path regularization loss
        pids_tensor = torch.LongTensor(pids).to(self.device)
        items = self.embedding["entity"].weight[: self.n_items]  # [bs, d]
        scores = torch.matmul(outputs[-1], items.t())  # [1, vocab_size]
        logprobs = F.log_softmax(scores, dim=1)  # [1, vocab_size]
        pid_logprobs = logprobs[0][pids_tensor]
        x = pid_logprobs.detach().cpu().numpy().tolist()
        del uid_tensor
        del pids_tensor
        return x

    def infer_with_path(self, metapath, uid, kg_mask, excluded_pids, topk_paths):
        """Reasoning paths over kg."""
        modules = self._get_modules(metapath)
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        # list of tensor of [1, d]
        outputs = self._forward(modules, uid_tensor)

        layer_logprobs = []
        for i, module in enumerate(modules):
            et_vecs = self.embedding[module.et_name]
            scores = torch.matmul(outputs[i], et_vecs.t())  # [1, vocab_size]
            logprobs = F.log_softmax(scores[0], dim=0)  # [vocab_size, ]
            layer_logprobs.append(logprobs)

        # Decide adaptive sampling size.
        num_valid_ids = len(kg_mask.get_ids("user", uid, self.relation2rid[modules[0].name]))
        if num_valid_ids <= 0:
            return []
        sample_sizes = [topk_paths, 5, 1]

        result_paths = [([uid], [])]  # (list of ids, list of scores)
        for i, module in enumerate(modules):  # iterate over each level
            tmp_paths = []
            visited_ids = []
            for path, value in result_paths:  # both are lists
                # Find valid node ids that are unvisited and not excluded pids.
                valid_et_ids = kg_mask.get_ids(module.eh_name, path[-1], self.relation2rid[module.name])
                valid_et_ids = set(valid_et_ids).difference(visited_ids)
                if i == len(modules) - 1 and excluded_pids is not None:
                    valid_et_ids = valid_et_ids.difference(excluded_pids)
                if len(valid_et_ids) <= 0:
                    continue
                valid_et_ids = list(valid_et_ids)

                # Compute top k nodes.
                valid_et_ids = torch.LongTensor(valid_et_ids).to(self.device)
                valid_et_logprobs = layer_logprobs[i].index_select(0, valid_et_ids)
                k = min(sample_sizes[i], len(valid_et_ids))
                topk_et_logprobs, topk_idxs = valid_et_logprobs.topk(k)
                topk_et_ids = valid_et_ids.index_select(0, topk_idxs)

                # Add nodes to path separately.
                topk_et_ids = topk_et_ids.detach().cpu().numpy()
                topk_et_logprobs = topk_et_logprobs.detach().cpu().numpy()
                for j in range(topk_et_ids.shape[0]):
                    new_path = path + [topk_et_ids[j]]
                    new_value = value + [topk_et_logprobs[j]]
                    tmp_paths.append((new_path, new_value))
                    # Remember to add the node to visited list!!!
                    visited_ids.append(topk_et_ids[j])
                del valid_et_ids
            if len(tmp_paths) <= 0:
                return []
            result_paths = tmp_paths
        del uid_tensor
        return result_paths


class RelationModule(nn.Module):
    def __init__(self, embedding_size, relation_info):
        super().__init__()
        self.name = relation_info["name"]
        self.eh_name = relation_info["entity_head"]
        self.et_name = relation_info["entity_tail"]
        self.fc1 = nn.Linear(embedding_size * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        """Compute log probability of output entity.
        Args:
            x: a FloatTensor of size [bs, input_size].
        Returns:
            FloatTensor of log probability of size [bs, output_size].
        """
        eh_vec, user_vec = inputs
        x = torch.cat([eh_vec, user_vec], dim=-1)
        x = self.bn1(self.dropout(F.relu(self.fc1(x))))
        out = self.fc2(x) + eh_vec
        return out


class DeepRelationModule(nn.Module):
    def __init__(self, embedding_size, relation_info, use_dropout):
        super().__init__()
        self.name = relation_info["name"]
        self.eh_name = relation_info["entity_head"]
        self.et_name = relation_info["entity_tail"]
        input_size = embedding_size * 2
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.fc3 = nn.Linear(input_size, embedding_size)
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, inputs):
        eh_vec, user_vec = inputs
        feature = torch.cat([eh_vec, user_vec], dim=-1)
        x = F.relu(self.fc1(feature))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.fc2(x) + feature)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn2(x)
        out = self.fc3(x)
        return out


class ReplayMemory:
    def __init__(self, memory_size=5000):
        self.memory_size = memory_size
        self.memory = []

    def add(self, data):
        # `data` is a list of objects.
        self.memory.extend(data)
        while len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample(self):
        # memory is empty.
        if not self.memory:
            return None
        return random.choice(self.memory)

    def __len__(self):
        return len(self.memory)


class KGMask:
    def __init__(self, kg, ui_relation_id):
        self.kg = kg
        self.ui_relation_id = ui_relation_id

    def _get_next_node_type(self, current_node_type, relation_id):
        if current_node_type == "entity" and relation_id == self.ui_relation_id:
            return "user"
        else:
            return "entity"

    def get_ids(self, eh, eh_ids, relation):
        et_ids = []
        if isinstance(eh_ids, list):
            for eh_id in eh_ids:
                try:
                    ids = list(self.kg[eh][eh_id][relation])
                except KeyError:
                    ids = []
                et_ids.extend(ids)
            et_ids = list(set(et_ids))
        else:
            try:
                res = self.kg[eh][eh_ids][relation]
            except KeyError:
                res = []
            et_ids = list(res)
        return et_ids

    def get_mask(self, eh, eh_ids, relation):
        et = self._get_next_node_type(eh, relation)
        et_vocab_size = len(self.kg[et])

        if isinstance(eh_ids, list):
            mask = np.zeros([len(eh_ids), et_vocab_size], dtype=np.int64)
            for i, eh_id in enumerate(eh_ids):
                try:
                    et_ids = list(self.kg[eh][eh_id][relation])
                except KeyError:
                    et_ids = []
                mask[i, et_ids] = 1
        else:
            mask = np.zeros(et_vocab_size, dtype=np.int64)
            try:
                et_ids = list(self.kg[eh][eh_ids][relation])
            except KeyError:
                et_ids = []
            mask[et_ids] = 1
        return mask

    def __call__(self, eh, eh_ids, relation):
        return self.get_mask(eh, eh_ids, relation)


class MetaProgramExecutor:
    """This implements the profile-guided reasoning algorithm."""

    def __init__(self, symbolic_model, random_generator, device, kg_mask, relation2rid):
        self.symbolic_model = symbolic_model
        self.kg_mask = kg_mask
        self.device = device
        self.relation2rid = relation2rid
        self.rng = random_generator

    def _get_module(self, relation):
        return getattr(self.symbolic_model, relation)

    def execute(self, program, uid, excluded_pids=None, adaptive_topk=False, manual_topk=5):
        """Execute the program to generate node representations and real nodes.
        Args:
            program: an instance of MetaProgram.
            uid: user ID (integer).
            excluded_pids: list of item IDs (list).
        """
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        user_vec = self.symbolic_model.embedding["user"][uid_tensor]  # tensor [1, d]
        root = program.root  # TreeNode
        root.data["vec"] = user_vec  # tensor [1, d]
        root.data["paths"] = [([uid], [], [], [])]  # (path, value, mp)

        excluded_pids = [] if excluded_pids is None else excluded_pids.tolist()

        # Run BFS to traverse tree.
        queue = root.get_children()
        while queue:  # queue is not empty
            node = queue.pop(0)
            child_nodes = self.rng.permutation(node.get_children())
            queue.extend(child_nodes)

            # Compute estimated vector of the node.
            x = (node.parent.data["vec"], user_vec)
            node.data["vec"] = self._get_module(node.relation)(x)  # tensor [1, d]

            # Compute scores (log prob) for the node.
            entity_vecs = self.symbolic_model.embedding[node.entity]  # tensor [vocab, d]
            # tensor [1, vocab]
            scores = torch.matmul(node.data["vec"], entity_vecs.t())
            scores = F.log_softmax(scores[0], dim=0)  # tensor [vocab, ]

            node.data["paths"] = []
            visited_ids = []
            for path, value, ep, mp in node.parent.data["paths"]:
                # Find valid node ids for current path.
                valid_ids = self.kg_mask.get_ids(node.parent.entity, path[-1], self.relation2rid[node.relation])
                valid_ids = set(valid_ids).difference(visited_ids)
                if not node.has_children() and excluded_pids:
                    valid_ids = valid_ids.difference(excluded_pids)
                if not valid_ids:  # empty list
                    continue
                valid_ids = list(valid_ids)

                # Compute top k nodes.
                valid_ids = torch.LongTensor(valid_ids).to(self.device)
                valid_scores = scores.index_select(0, valid_ids)
                if adaptive_topk:
                    k = min(node.sample_size, len(valid_ids))
                else:
                    k = min(manual_topk, len(valid_ids))
                topk_scores, topk_idxs = valid_scores.topk(k)
                topk_ids = valid_ids.index_select(0, topk_idxs)

                # Add nodes and scores to paths.
                topk_ids = topk_ids.detach().cpu().numpy()
                topk_scores = topk_scores.detach().cpu().numpy()
                for j in range(k):
                    new_path = path + [topk_ids[j]]
                    new_value = value + [topk_scores[j]]
                    new_mp = mp + [node.relation]
                    new_ep = ep + [node.entity]
                    node.data["paths"].append((new_path, new_value, new_ep, new_mp))

                    # Remember to add the node to visited list!!!
                    visited_ids.append(topk_ids[j])
                    if not node.has_children():
                        excluded_pids.append(topk_ids[j])

    def collect_results(self, program):
        results = []
        queue = program.root.get_children()
        while len(queue) > 0:
            node = queue.pop(0)
            queue.extend(node.get_children())
            if not node.has_children():
                results.extend(node.data["paths"])
        return results


class NeuralProgramLayout:
    """This refers to the layout tree in the paper."""

    def __init__(self, metapaths):
        super().__init__()
        self.mp2id = {}
        for mpid, mp in enumerate(metapaths):
            simple_mp = tuple([v[0] for v in mp[1:]])
            self.mp2id[simple_mp] = mpid

        self.root = TreeNode(0, "user", None)
        for mp in metapaths:
            node = self.root
            for i in range(1, len(mp)):
                if mp[i] not in node.children:
                    node.children[mp[i]] = TreeNode(i, mp[i][1], mp[i][0])
                    node.children[mp[i]].parent = node
                node = node.children[mp[i]]

    def update_by_path_count(self, path_count):
        """Update sample size of each node by expected number of paths.
        Args:
            path_count: dict with key=mpid, value=int
        """

        def _postorder_update(node, parent_rels):
            if not node.has_children():
                mpid = self.mp2id[tuple(parent_rels)]
                node.sample_size = int(path_count[mpid])
                return

            min_pos_sample_size, max_sample_size = 99, 0
            for child in node.get_children():
                _postorder_update(child, parent_rels + [child.relation])
                max_sample_size = max(max_sample_size, child.sample_size)
                if child.sample_size > 0:
                    min_pos_sample_size = min(min_pos_sample_size, child.sample_size)

            # Update current node sampling size.
            # a) if current node is root, set to 1.
            if not node.has_parent():
                node.sample_size = 1
            # b) if current node is not root, and all children sample sizes are 0, set to 0.
            elif max_sample_size == 0:
                node.sample_size = 0
            # c) if current node is not root, take the minimum and update children.
            else:
                node.sample_size = min_pos_sample_size
                for child in node.get_children():
                    child.sample_size = int(child.sample_size / node.sample_size)

        _postorder_update(self.root, [])

    def print_postorder(self, hide_branch=True):
        def _postorder(node, msgs):
            msg = (node.entity, node.relation, node.sample_size)
            new_msgs = msgs + [msg]

            if not node.has_children():
                if hide_branch and msg[2] == 0:
                    return
                str_msgs = [f"({msg[0]},{msg[1]},{msg[2]})" for msg in new_msgs]
                print("  ".join(str_msgs))
                return

            for child in node.children:
                _postorder(child, new_msgs)

        _postorder(self.root, [])


class TreeNode:
    def __init__(self, level, entity, relation):
        super().__init__()
        self.level = level
        self.entity = entity  # Entity type
        self.relation = relation  # Relation pointing to this tail entity
        self.parent = None
        self.children = {}  # key = (relation, entity), value = TreeNode
        self.sample_size = 0  # number of nodes to sample
        self.data = {}  # extra information to save

    def has_parent(self):
        return self.parent is not None

    def has_children(self):
        return len(self.children) > 0

    def get_children(self):
        return list(self.children.values())

    def __str__(self):
        parent = None if not self.has_parent() else self.parent.entity
        msg = f"({parent},{self.relation},{self.entity})"
        return msg
