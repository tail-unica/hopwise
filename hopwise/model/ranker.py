# @Time   : 2025/06
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.model.ranker
#######################
Common ranker in recommender system
"""

from collections import defaultdict

import torch

from hopwise.utils import PathLanguageModelingTokenType


class RankerLP:
    def __init__(self, tokenizer, kg_positives, K=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.kg_positives = kg_positives
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)
        self.max_new_tokens = max_new_tokens
        self.K = K

    def update_topk(self, generate_outputs):
        sorted_scores = generate_outputs.sequences_scores.argsort(descending=True)
        generate_outputs.sequences = generate_outputs.sequences[sorted_scores]
        for sequence in generate_outputs.sequences:
            seq = self.tokenizer.decode(sequence).split(" ")
            head_eid = int(seq[1][1:])
            rel_rid = int(seq[2][1:])
            if len(self.topk[head_eid, rel_rid]) >= self.K:
                continue
            recommended_token = seq[-1]
            recommended_item = int(recommended_token[1:])
            if (
                recommended_item in self.kg_positives[(head_eid, rel_rid)]
                or recommended_item in self.topk[head_eid, rel_rid]
            ):
                continue
            self.topk[head_eid, rel_rid].append(recommended_item)
            self.topk_sequences[head_eid, rel_rid].append(seq)

    def reset_topks(self):
        del self.topk
        del self.topk_sequences
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)


class CumulativeSequenceScoreRanker:
    """
    Ranker that uses the cumulative sequence score of the final 5 predicted tokens to rank sequences.
    """

    def __init__(self, tokenizer, used_ids, item_num, topk=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.used_ids = used_ids
        self.item_num = item_num
        self.max_new_tokens = max_new_tokens
        self.topk = topk
        self.avg_topk_size = {}

    def calculate_sequence_scores(self, normalized_tuple, sequences):
        new_sequence_tokens = sequences[:, -self.max_new_tokens - 1 : -1]
        sequence_scores = []
        # Iterate over each tensor in the normalized tuple
        for i in range(self.max_new_tokens):
            # Get the probabilities corresponding to the ith token in new_sequence_tokens
            probs = normalized_tuple[i].gather(1, new_sequence_tokens[:, [i]])
            sequence_scores.append(probs)
        # Convert the list of tensors into a single tensor
        sequence_scores = torch.cat(sequence_scores, dim=-1)
        # Calculate the average score over the last 5 positions for each sequence
        sequence_scores = sequence_scores.mean(dim=-1)
        return sequence_scores

    def normalize_tuple(self, logits_tuple):
        # Normalize each tensor in the tuple
        normalized_tuple = tuple(torch.softmax(logits, dim=-1) for logits in logits_tuple)
        return normalized_tuple

    def get_sequences(self, batch_len, generation_outputs):
        user_num = batch_len
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        normalized_scores = self.normalize_tuple(generation_outputs.scores)
        normalized_sequences_scores = self.calculate_sequence_scores(normalized_scores, generation_outputs.sequences)

        sequences = generation_outputs.sequences
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        valid_sequences_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))  # false if finite
        normalized_sequences_scores = torch.where(valid_sequences_mask, -torch.inf, normalized_sequences_scores)

        sorted_indices = normalized_sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_sequences_scores = normalized_sequences_scores[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]

        for user_index, sequence, sequence_score in zip(
            sorted_batch_user_index, sorted_sequences, sorted_sequences_scores
        ):
            seq = self.tokenizer.decode(sequence).split(" ")

            uid_token = seq[1]
            recommended_token = seq[-1]

            if recommended_token == self.tokenizer.pad_token:
                continue

            if not (
                uid_token.startswith(PathLanguageModelingTokenType.USER.value)
                and recommended_token.startswith(PathLanguageModelingTokenType.ITEM.value)
            ):
                continue

            uid = int(uid_token[1:])
            recommended_item = int(recommended_token[1:])

            if torch.isfinite(scores[user_index, recommended_item]) or recommended_item in self.used_ids[uid]:
                continue
            scores[user_index, recommended_item] = sequence_score

            if uid not in self.avg_topk_size:
                self.avg_topk_size[uid] = set()

            user_topk_sequences.append((uid, recommended_item, scores[user_index, recommended_item].item(), seq))
            self.avg_topk_size[uid].add(seq[-1])

        return scores, user_topk_sequences


class SampleSearchSequenceScoreRanker:
    """
    Ranker that uses the sequence score of the beam search to rank sequences.

    To use only if do_sample = True and if topk and topp are set.
    """

    def __init__(self, tokenizer, used_ids, item_num, topk=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.used_ids = used_ids
        self.item_num = item_num
        self.topk = topk
        self.max_new_tokens = max_new_tokens
        self.avg_topk_size = {}

    def get_scores(self, sequences, scores):
        sequences_scores = None

        for i, tstep in enumerate(scores):
            # tstep is a tensor for logits at time t
            score = torch.softmax(tstep, dim=-1)
            if sequences_scores is None:
                sequences_scores = score[:, sequences[:, i]].sum(-1)
            else:
                sequences_scores += score[:, sequences[:, i]].sum(-1)

        return sequences_scores

    def get_sequences(self, batch_len, generation_outputs):
        """
        generation_outputs is a dict with 3 keys: 'sequences', 'scores' and 'past_key_values'
        sequences is a tensor of shape (num_return_sequences, sequence_length)
        scores is a tuple of len (|generated tokens|) where each element is a tensor
            that says the logits at each timestep before applying topk and topp

        """
        user_num = batch_len
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        sequences = generation_outputs.sequences
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        sequences_score = self.get_scores(sequences[:, -self.max_new_tokens :], generation_outputs.scores)
        for sequence, user_index, sequence_score in zip(sequences, batch_user_index, sequences_score):
            seq = self.tokenizer.decode(sequence).split(" ")
            uid_token = seq[1]
            recommended_token = seq[-1]

            if recommended_token == self.tokenizer.pad_token:
                continue

            if not (
                uid_token.startswith(PathLanguageModelingTokenType.USER.value)
                and recommended_token.startswith(PathLanguageModelingTokenType.ITEM.value)
            ):
                continue

            uid = int(uid_token[1:])
            recommended_item = int(recommended_token[1:])

            if torch.isfinite(scores[user_index, recommended_item]) or recommended_item in self.used_ids[uid]:
                continue
            scores[user_index, recommended_item] = sequence_score

            if uid not in self.avg_topk_size:
                self.avg_topk_size[uid] = set()

            user_topk_sequences.append((uid, recommended_item, scores[user_index, recommended_item].item(), seq))
            self.avg_topk_size[uid].add(seq[-1])

        return scores, user_topk_sequences


class BeamSearchSequenceScoreRanker:
    """
    Ranker that uses the sequence score of the beam search to rank sequences.
    """

    def __init__(self, tokenizer, used_ids, item_num, topk=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.used_ids = used_ids
        self.item_num = item_num
        self.topk = topk
        self.avg_topk_size = {}

    def get_sequences(self, batch_len, generation_outputs):
        user_num = batch_len
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        sequences = generation_outputs.sequences
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        sorted_indices = generation_outputs.sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]

        for sequence, user_index, sequence_score in zip(sorted_sequences, sorted_batch_user_index, sorted_indices):
            seq = self.tokenizer.decode(sequence).split(" ")
            uid_token = seq[1]
            recommended_token = seq[-1]

            if not (
                uid_token.startswith(PathLanguageModelingTokenType.USER.value)
                and recommended_token.startswith(PathLanguageModelingTokenType.ITEM.value)
            ):
                continue

            uid = int(uid_token[1:])
            recommended_item = int(recommended_token[1:])

            if torch.isfinite(scores[user_index, recommended_item]) or recommended_item in self.used_ids[uid]:
                continue
            scores[user_index, recommended_item] = sequence_score
            if uid not in self.avg_topk_size:
                self.avg_topk_size[uid] = set()

            user_topk_sequences.append((uid, recommended_item, scores[user_index, recommended_item].item(), seq))
            self.avg_topk_size[uid].add(seq[-1])

        return scores, user_topk_sequences
