# @Time   : 2025/05/28
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""TPRec
##################################################
Reference: Time-aware Path Reasoning on Knowledge Graph for Recommendation (https://arxiv.org/pdf/2108.02634)

"""

from collections import defaultdict, namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from hopwise.model.abstract_recommender import ExplainableRecommender, KnowledgeRecommender
from hopwise.utils import InputType


class TPRec(KnowledgeRecommender, ExplainableRecommender):
    """
    TPRec

    1. Train TransE embeddings and preprocess it according to the preprocess embedding notebook
    2. Run TPRec with 'pretrain' train stage set
    3. Run TPRec with 'policy' train stage set

    """

    input_type = InputType.USERWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config

        # Load parameters info from config
        self.user_num = dataset.user_num
        self.device = config["device"]
        self.topk = config["topk"]

        # PGPR Configurations
        self.state_history = config["state_history"]
        self.max_acts = config["max_acts"]
        self.gamma = config["gamma"]
        self.action_dropout = config["action_dropout"]
        self.hidden_sizes = config["hidden_sizes"]
        self.act_dim = self.max_acts + 1
        self.max_num_nodes = config["max_path_len"] + 1
        self.weight_factor = config["weight_factor"]
        self.path_pattern = config["path_constraint"]
        self.beam_search_hop = config["beam_search_hop"]
        self.train_stage = config["train_stage"]
        self.margin = config["margin"]
        self.n_clusters = config["cluster_num"]
        self.fix_scores_sorting_bug = config["fix_scores_sorting_bug"]

        # user-item relation
        self.ui_relation = dataset.ui_relation
        self.ui_relation_id = dataset.field2token_id["relation_id"][dataset.ui_relation]

        self.graph_dict = dataset.ckg_dict_graph()
        # Items
        self.positives = dataset.history_item_matrix()[0]
        self.pretrained_weights = {}  # during pretraining is None, otherwise it contain pretrained weights
        self.uc_weight = dataset.temporal_weights.uc_weight
        self.timenum = dataset.temporal_weights.timenum
        self.ui2label_dict = dataset.temporal_weights.timeClassifyLabel

        # Load Knowledge Graph Embedding Checkpoint
        if self.train_stage == "pretrain":
            # then load transe embeddings and initialize new torch embeddings
            self.user_embedding = dataset.get_preload_weight("user_embedding_id")
            self.entity_embedding = dataset.get_preload_weight("entity_embedding_id")
            self.relation_embedding = dataset.get_preload_weight("relation_embedding_id")

            # make embeddings learnable
            self.user_embedding = torch.from_numpy(self.user_embedding)
            self.entity_embedding = torch.from_numpy(self.entity_embedding)
            self.relation_embedding = torch.from_numpy(self.relation_embedding)

            self.user_embedding = nn.Embedding.from_pretrained(self.user_embedding)
            self.entity_embedding = nn.Embedding.from_pretrained(self.entity_embedding)
            self.relation_embedding = nn.Embedding.from_pretrained(self.relation_embedding)

            # TPRec temporal embeddings
            embedding_size = self.user_embedding.weight.size(1)
            self.ui_clust_relation_embedding = nn.Embedding(self.n_clusters, embedding_size)

            self.transe_loss = nn.TripletMarginLoss(margin=config["margin"], p=2, reduction="mean")
            return
        else:
            # load pretrained weights
            pretrained_weights = self._get_pretrained_weights()
            # then transform torch embedding in numpy
            # transform torch.load self.pretrained_weights in numpy
            self.user_embedding = pretrained_weights["user_embedding.weight"].cpu().numpy()
            self.entity_embedding = pretrained_weights["entity_embedding.weight"].cpu().numpy()
            self.relation_embedding = pretrained_weights["relation_embedding.weight"].cpu().numpy()
            self.ui_clust_relation_embedding = pretrained_weights["ui_clust_relation_embedding.weight"].cpu().numpy()
        self.embedding_size = self.user_embedding.shape[1]
        self.state_gen = KGState(self.embedding_size, self.state_history)

        # Actor-Critic model
        self.l1 = nn.Linear(self.state_gen.dim, self.hidden_sizes[0])
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.actor = nn.Linear(self.hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(self.hidden_sizes[1], 1)

        # Self Loop Embedding
        self.self_loop_embedding = np.zeros(self.embedding_size)

        # Map Relation ID to relation name to check has pattern constraint
        self.rid2relation = {v: k for k, v in dataset.field2token_id["relation_id"].items()}

        # Mapping node type to embeddings
        self.node_type2emb = {
            "user": self.user_embedding,
            "entity": self.entity_embedding,
            "relation": self.relation_embedding,
            "self_loop": self.self_loop_embedding,
        }

        # Normalization score
        u_p_scores = np.dot(
            self.user_embedding + self.relation_embedding[self.ui_relation_id], self.entity_embedding[: self.n_items].T
        )
        self.u_p_scales = np.max(u_p_scores, axis=1)

        # These are the paths constraint to use when checking for path correctness in _get_reward function
        # Preprocess the path constraints.
        self.patterns = list()
        for path_constraint in self.path_pattern:
            relations = list()
            for node in path_constraint:
                path_rel = node[0]
                if path_rel is not None:
                    if path_rel.endswith("_r"):
                        # remove the reverse suffix
                        path_rel = path_rel[:-2]
                    relations.append(path_rel)
            self.patterns.append(tuple(["self_loop"]) + tuple(relations))

        # this second step is added separately for clarity
        # expand UI-Relation to the number of clusters according to path preprocessing in TPRec
        extended_paths = []
        for path_constraint in self.patterns.copy():
            extended_paths.extend(self.expand_paths(path_constraint))
        self.patterns = extended_paths  # remove duplicates

        # Following is current episode information.
        self._batch_path = None
        self._batch_curr_actions = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False
        self.saved_actions = []
        self.rewards = []
        self.entropy = []

        # random generator
        self.rng = np.random.default_rng()
        self.SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

    def expand_paths(self, path_constraint):
        """
        Expand the path constraint by replacing the ui_relation with multiple clusters
        like ui_relation0 to ui_relation_n with n being the number of clusters.

        Args:
            path_constraint in the form of [rel1, rel2, rel3]
        """

        def _expand_recursive(current_path, remaining_relations):
            # Base case - path complete
            if not remaining_relations:
                return [current_path]

            # Get next relation
            relation = remaining_relations[0]

            # If ui_relation, branch into n_clusters paths
            if relation == self.ui_relation:
                expanded = []
                for i in range(self.n_clusters):
                    cluster_path = current_path + [f"{relation}_{i}"]
                    expanded.extend(_expand_recursive(cluster_path, remaining_relations[1:]))
                return expanded
            else:
                # Regular relation - add and continue
                return _expand_recursive(current_path + [relation], remaining_relations[1:])

        # Start expansion from empty path
        return _expand_recursive([], path_constraint[1:])  # Skip self-loop

    def _get_pretrained_weights(self):
        import os

        checkpoint_file = os.path.join(
            self.config["checkpoint_dir"],
            "{}-{}-{}.pth".format(self.config["model"], self.config["dataset"], "pretrained"),
        )
        checkpoint = torch.load(checkpoint_file, weights_only=False, map_location=self.device)
        weights = checkpoint["state_dict"]
        return weights

    def select_action(self, batch_state, batch_act_mask):
        # Tensor [bs, state_dim]
        state = torch.FloatTensor(batch_state).to(self.device)
        # Tensor of [bs, act_dim]
        act_mask = torch.BoolTensor(batch_act_mask).to(self.device)
        # act_probs: [bs, act_dim], state_value: [bs, 1]
        probs, value = self.forward((state, act_mask))
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(self.SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self):  # prev update
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        # numpy array of [bs, #steps]
        batch_rewards = np.vstack(self.rewards).T
        batch_rewards = torch.tensor(batch_rewards).to(self.device)
        num_steps = batch_rewards.shape[1]

        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0

        for i in range(0, num_steps):
            # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            log_prob, value = self.saved_actions[i]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]

        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + self.weight_factor * entropy_loss

        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss, actor_loss, critic_loss, entropy_loss

    def forward(self, inputs):
        # used only in inference
        # state: [bs, state_dim], act_mask: [bs, act_dim]
        state, act_mask = inputs
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)
        actor_logits = self.actor(x)
        actor_logits[~act_mask] = float("-inf")
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]
        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def _get_transe_rec_embedding(self, user, pos_item, neg_item):
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return user_e, pos_item_e, neg_item_e, rec_r_e

    def _get_transe_kg_embedding(self, head, pos_tail, neg_tail, relation):
        head_e = self.entity_embedding(head)
        pos_tail_e = self.entity_embedding(pos_tail)
        neg_tail_e = self.entity_embedding(neg_tail)
        relation_e = self.relation_embedding(relation)
        return head_e, pos_tail_e, neg_tail_e, relation_e

    def calculate_loss_transe(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        ui_clusters_nums = [
            self.ui2label_dict[(user.item(), pos_item.item())] for user, pos_item in zip(user, pos_item)
        ]
        num_id, num_index, num_n = np.unique(ui_clusters_nums, return_index=True, return_counts=True)
        neg_item = interaction[self.NEG_ITEM_ID]
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]

        user_e, pos_item_e, neg_item_e, rec_r_e = self._get_transe_rec_embedding(user, pos_item, neg_item)
        head_e, pos_tail_e, neg_tail_e, relation_e = self._get_transe_kg_embedding(head, pos_tail, neg_tail, relation)

        loss = 0
        if len(num_id) == 1:
            rec_r_e = self.ui_clust_relation_embedding.weight[num_id[0]]
        else:
            tmp_vec_rel = None
            for i, cur_cls in enumerate(num_id):
                startID = num_index[i]
                endID = num_index[i] + num_n[i]
                rec_clus_r_e = self.ui_clust_relation_embedding.weight[cur_cls]
                if tmp_vec_rel is None:
                    tmp_vec_rel = rec_r_e
                else:
                    torch.cat((rec_r_e[0], tmp_vec_rel[0]), 0)
                loss += self.transe_loss(
                    user_e[startID:endID] + rec_clus_r_e, pos_item_e[startID:endID], neg_item_e[startID:endID]
                )
        h_e = torch.cat([user_e, head_e])
        r_e = torch.cat([rec_r_e, relation_e])
        pos_t_e = torch.cat([pos_item_e, pos_tail_e])
        neg_t_e = torch.cat([neg_item_e, neg_tail_e])

        loss += self.transe_loss(h_e + r_e, pos_t_e, neg_t_e)

        return loss

    def forward_transe(self, user, relation, item):
        score = -torch.norm(user + relation - item, p=2, dim=1)
        return score

    def predict_transe(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)
        item_e = self.entity_embedding(item)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        return self.forward_transe(user_e, rec_r_e, item_e)

    def full_sort_predict_transe(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)

        rec_r_e = self.relation_embedding.weight[-1]
        rec_r_e = rec_r_e.expand_as(user_e)

        item_indices = torch.tensor(range(self.n_items)).to(self.device)
        all_item_e = self.entity_embedding.weight[item_indices]

        user_e = user_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        rec_r_e = rec_r_e.unsqueeze(1).expand(-1, all_item_e.shape[0], -1)
        t = all_item_e.unsqueeze(0)

        return -torch.norm(user_e + rec_r_e - t, p=2, dim=2)

    def calculate_loss(self, interaction):
        if self.train_stage == "pretrain":
            return self.calculate_loss_transe(interaction)

        users = interaction[self.USER_ID]
        users = users[users != 0]

        # Policy training
        batch_state = self.reset(users)
        done = False
        while not done:
            batch_act_mask = self.batch_action_mask(dropout=self.action_dropout)
            batch_act_idx = self.select_action(batch_state, batch_act_mask)
            batch_state, batch_reward, done = self.batch_step(batch_act_idx)
            self.rewards.append(batch_reward)
        loss, ploss, vloss, eloss = self.update()

        return loss

    def _has_pattern(self, path):
        pattern = tuple([self.rid2relation[v[0]] if v[0] != "self_loop" else v[0] for v in path])
        return pattern in self.patterns

    def _get_next_node_type(self, current_node_type, relation_id):
        if current_node_type == "entity" and relation_id == self.ui_relation_id:
            return "user"
        else:
            return "entity"

    def reset(self, user):
        self._batch_path = [[("self_loop", "user", uid)] for uid in user]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        return self._batch_curr_state

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_actions(self, path, done):
        # Compute actions for current node.
        _, curr_node_type, curr_node_id = path[-1]
        actions = [("self_loop", curr_node_id)]

        # (1) If game is finished, only return self-loop action.
        if done:
            return actions

        # (2) Get all possible edges from original knowledge graph.
        # [CAVEAT] Must remove visited nodes!
        if isinstance(curr_node_id, torch.Tensor):
            curr_node_id = curr_node_id.item()

        try:
            relations_nodes = self.graph_dict[curr_node_type][curr_node_id]
        except KeyError:
            relations_nodes = []
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])

        for r in relations_nodes:
            next_node_type = self._get_next_node_type(curr_node_type, r)
            next_node_ids = relations_nodes[r]
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        # (3) If candidate action set is empty, only return self-loop action.
        if len(candidate_acts) == 0:
            return actions

        # (4) If number of available actions is smaller than max_acts, return action sets.
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        # (5) If there are too many actions, do some deterministic trimming here!
        uid = path[0][-1]
        user_embed = self.user_embedding[uid]

        scores = []
        item_emb = None
        if isinstance(uid, torch.Tensor):
            uid = uid.item()

        for clus_wt in self.uc_weight[uid]:
            if item_emb is None:
                item_emb = self.ui_clust_relation_embedding[clus_wt] * self.uc_weight[uid][clus_wt]
            else:
                item_emb += self.ui_clust_relation_embedding[clus_wt] * self.uc_weight[uid][clus_wt]

        for r, next_node_id in candidate_acts:
            next_node_type = self._get_next_node_type(curr_node_type, r)
            if next_node_type == "user":
                src_embed = user_embed
            elif next_node_type == "entity" and next_node_id < self.n_items:
                src_embed = user_embed + item_emb
            else:
                src_embed = user_embed + item_emb + self.relation_embedding[r]

            score = np.matmul(src_embed, self.node_type2emb[next_node_type][next_node_id])
            # This trimming may filter out target items!
            # Manually set the score of target items a very large number.
            # if next_node_type == ITEM and next_node_id in self._target_pids:
            #    score = 99999.0
            scores.append(score)

        # choose actions with larger scores
        candidate_idxs = np.argsort(scores)[-self.max_acts :]
        if self.fix_scores_sorting_bug:
            candidate_acts = [candidate_acts[i] for i in candidate_idxs[::-1]]
        else:
            candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_state(self, batch_path):
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)  # [bs, dim]

    def _get_state(self, path):
        # Return state of torch vector: [user_embed, curr_node_embed, last_node_embed, last_relation].
        user_embed = self.user_embedding[path[0][-1]]
        zero_embed = np.zeros(self.embedding_size)

        if len(path) == 1:  # initial state
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]

        curr_node_embed = self.node_type2emb[curr_node_type][curr_node_id]
        last_node_embed = self.node_type2emb[last_node_type][last_node_id]

        if last_relation != "self_loop":
            last_relation_embed = self.relation_embedding[last_relation]
        else:
            last_relation_embed = self.self_loop_embedding

        if len(path) == 2:  # noqa: PLR2004
            state = self.state_gen(
                user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed, zero_embed
            )
            return state

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.node_type2emb[older_node_type][older_node_id]

        if older_relation == "self_loop":
            older_relation_embed = self.self_loop_embedding
        else:
            older_relation_embed = self.relation_embedding[older_relation]

        state = self.state_gen(
            user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed
        )
        return state

    def _batch_get_reward(self, batch_path):
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _get_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) <= 2:  # noqa: PLR2004
            return 0.0

        if not self._has_pattern(path):
            return 0.0

        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]

        if curr_node_type == "entity" and curr_node_id < self.n_items:
            # Give soft reward for other reached items.
            uid = path[0][-1]
            item_emb = None
            for clus_wt in self.uc_weight[uid.item()]:
                if item_emb is None:
                    item_emb = self.ui_clust_relation_embedding[clus_wt] * self.uc_weight[uid.item()][clus_wt]
                else:
                    item_emb += self.ui_clust_relation_embedding[clus_wt] * self.uc_weight[uid.item()][clus_wt]

            u_vec = self.user_embedding[uid] + item_emb
            p_vec = self.entity_embedding[curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            target_score = max(score, 0.0)
        return target_score

    def _is_done(self):
        # Episode ends only if max path length is reached.
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def batch_step(self, batch_act_idx):
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, _ = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]

            if relation == "self_loop":
                next_node_type = curr_node_type
            else:
                next_node_type = self._get_next_node_type(curr_node_type, relation)
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout):
        # Return action masks of size [bs, act_dim].
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = np.arange(len(actions))
            if dropout > 0 and len(act_idxs) >= 5:  # noqa: PLR2004
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = self.rng.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = np.concatenate([[act_idxs[0]], tmp])
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def _batch_acts_to_masks(self, batch_acts):
        batch_masks = np.zeros((len(batch_acts), self.act_dim), dtype=np.uint8)
        for i, acts in enumerate(batch_acts):
            num_acts = len(acts)
            batch_masks[i, :num_acts] = 1
        return batch_masks

    def predict(self, interaction):
        if self.train_stage == "pretrain":
            return self.predict_transe(interaction)
        return

    def full_sort_predict(self, interaction):
        if self.train_stage == "pretrain":
            return self.full_sort_predict_transe(interaction)

        # get set temporal weights
        interaction, temporal_weight = interaction
        users = interaction[self.USER_ID]
        paths, probs = self.beam_search(users)
        interacted_matrix = self._build_interacted_matrix(temporal_weight)
        return self.collect_scores(users, paths, probs, interacted_matrix)

    def _build_interacted_matrix(self, temporal_weight):
        item_emb = None
        purchase_matrix = []
        for uid in range(1, len(temporal_weight.uc_weight) + 1):
            for clus_wt in temporal_weight.uc_weight[uid]:
                if item_emb is None:
                    item_emb = self.ui_clust_relation_embedding[clus_wt] * temporal_weight.uc_weight[uid][clus_wt]
                else:
                    item_emb += self.ui_clust_relation_embedding[clus_wt] * temporal_weight.uc_weight[uid][clus_wt]
            purchase_matrix.append(item_emb)

        return purchase_matrix

    def explain(self, interaction):
        """Support function used for case study.

        Args:
            interaction : test interaction data

        Returns:
            pd.Dataframe: explanation results with columns: "user", "item", "score", "path"
        """
        users, temporal_weight = interaction[self.USER_ID]
        paths, probs = self.beam_search(users)
        interacted_matrix = self._build_interacted_matrix(temporal_weight)

        scores, explanations = self.collect_scores(users, paths, probs, interacted_matrix)

        for exp in explanations:
            exp[-1] = self.decode_path(exp[-1])

        return scores, explanations

    def decode_path(self, path):
        return path

    def beam_search(self, users):
        users = [user.item() for user in users]
        state_pool = self.reset(users)  # numpy of [bs, dim]
        path_pool = self._batch_path  # list of list, size=bs
        probs_pool = [[] for _ in users]

        for hop, k in enumerate(self.beam_search_hop):
            state_tensor = torch.FloatTensor(state_pool).to(self.device)
            acts_pool = self._batch_get_actions(path_pool, False)  # list of list, size=bs
            actmask_pool = self._batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
            actmask_tensor = torch.BoolTensor(actmask_pool).to(self.device)
            # Tensor of [bs, act_dim]
            probs, _ = self.forward((state_tensor, actmask_tensor))
            # In order to differ from masked actions
            probs = probs + actmask_tensor.float()
            topk_probs, topk_idxs = torch.topk(probs, k, dim=1)  # LongTensor of [bs, k]

            topk_idxs = topk_idxs.detach().cpu().numpy()
            topk_probs = topk_probs.detach().cpu().numpy()

            new_path_pool, new_probs_pool = [], []
            for row in range(topk_idxs.shape[0]):
                path = path_pool[row]
                probs = probs_pool[row]
                for idx, p in zip(topk_idxs[row], topk_probs[row]):
                    if idx >= len(acts_pool[row]):  # act idx is invalid
                        continue

                    # (relation, next_node_id)
                    relation, next_node_id = acts_pool[row][idx]

                    if relation == "self_loop":
                        next_node_type = path[-1][1]
                    else:
                        next_node_type = self._get_next_node_type(path[-1][1], relation)

                    new_path = path + [(relation, next_node_type, next_node_id)]

                    new_path_pool.append(new_path)
                    new_probs_pool.append(probs + [p])
            path_pool = new_path_pool
            probs_pool = new_probs_pool
            if hop < 2:  # noqa: PLR2004
                state_pool = self._batch_get_state(path_pool)

        return path_pool, probs_pool

    def collect_scores(self, users, paths, probs, interacted_matrix):
        collect_results = list()
        pad_emb = np.zeros((1, self.embedding_size))
        interacted_embeds = np.concatenate((pad_emb, np.array(interacted_matrix)))
        # 1) get all valid paths for each user, compute path score and path probability
        pred_paths = {uid.item(): defaultdict(list) for uid in users}
        path_scores = np.dot(self.user_embedding + interacted_embeds, self.entity_embedding[: self.n_items].T)
        for path, prob in zip(paths, probs):
            if "self_loop" in [node[0] for node in path[1:]]:
                continue

            if path[-1][1] != "entity":
                continue

            path_uid = path[0][2]
            # check it is a user in the test set batch
            if path_uid not in pred_paths:
                continue

            path_pid = path[-1][2]
            # check it is an item
            if not (path_pid < self.n_items):
                continue

            if path_pid in self.positives[path_uid]:
                continue

            path_score = path_scores[path_uid][path_pid]
            path_prob = reduce(lambda x, y: x * y, prob)
            pred_paths[path_uid][path_pid].append((path_score, path_prob, path))

        # 2) Pick best paths for each user-item pair based on the score
        best_pred_paths = defaultdict(list)
        for user, user_pred_paths in pred_paths.items():
            for item in user_pred_paths:
                if item in self.positives[user]:
                    continue
                # Get the path with highest probability
                sorted_path = sorted(user_pred_paths[item], key=lambda x: x[1], reverse=True)[0]
                best_pred_paths[user].append(sorted_path)

        # 3) Fill the results tensor
        results = torch.full((len(users), self.n_items), -torch.inf)

        for i, user in enumerate(best_pred_paths):
            # sort by score
            sorted_path = sorted(best_pred_paths[user], key=lambda x: (x[0], x[1]), reverse=True)
            top_items = [[p[-1][2], score] for score, _, p in sorted_path][: max(self.topk)]
            top_paths = [p for _, _, p in sorted_path][: max(self.topk)]
            if len(top_items) < max(self.topk):
                cand_pids = np.argsort(path_scores[user])
                for cand_pids in cand_pids[::-1]:
                    if cand_pids in self.positives[user]:
                        continue
                    top_items.append([cand_pids, path_scores[user][cand_pids]])
                    if len(top_items) >= max(self.topk):
                        break
            # Change order from smallest to largest
            top_items = top_items[::-1]
            top_paths = top_paths[::-1]
            for (item, score), path in zip(top_items, top_paths):
                results[i, item] = score.tolist()

                # collect user, item, score and paths.
                collect_results.append([user, item, score, path])

        return results, collect_results


class KGState:
    def __init__(self, embedding_size, history_len):
        self.embedding_size = embedding_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embedding_size
        elif history_len == 1:
            self.dim = 4 * embedding_size
        elif history_len == 2:  # noqa: PLR2004
            self.dim = 6 * embedding_size
        else:
            raise Exception("history length should be one of {0, 1, 2}")

    def __call__(
        self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed
    ):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:  # noqa: PLR2004
            return np.concatenate(
                [user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed]
            )
        else:
            raise ValueError("mode should be one of {full, current}")
