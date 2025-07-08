# @Time   : 2025/06
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.model.postprocessor
#######################
Common post-processors for path sequences in path language modeling recommender systems.
"""

from collections import defaultdict

import torch

from hopwise.utils import PathLanguageModelingTokenType


class BaseSequencePostProcessor:
    """
    Base class for sequence score post-processors.
    """

    def __init__(self, tokenizer, used_ids, item_num, topk=10):
        self.tokenizer = tokenizer
        self.used_ids = used_ids
        self.item_num = item_num
        self.topk = topk

    def get_sequences(self, generation_outputs, max_new_tokens=24):
        """
        This method should be implemented by subclasses to extract sequences and their scores.

        Args:
            generation_outputs: A mapping containing the generated sequences and their scores.
            max_new_tokens: The maximum number of new tokens to consider for scoring.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def parse_sequences(self, user_index, sequences, sequences_scores):
        """
        Parses the sequences and scores to extract user IDs, recommended items, and their scores.

        Args:
            user_index (torch.Tensor): A tensor containing user indices.
            sequences (torch.Tensor): A tensor containing the generated sequences.
            sequences_scores (torch.Tensor): A tensor containing the scores for each sequence.

        Returns:
            scores (torch.Tensor): A tensor of shape (user_num, item_num) containing the scores for each user and item.
            user_topk_sequences (list): A list of lists, where each inner list contains:
                [user_id, recommended_item, sequence_score, decoded_sequence].
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        for batch_uidx, sequence, sequence_score in zip(user_index, sequences, sequences_scores):
            parsed_seq = self._parse_single_sequence(scores, batch_uidx, sequence)
            if parsed_seq is None:
                continue
            uid, recommended_item, decoded_seq = parsed_seq

            scores[batch_uidx, recommended_item] = sequence_score
            user_topk_sequences.append([uid, recommended_item, sequence_score.item(), decoded_seq])

        return scores, user_topk_sequences

    def _parse_single_sequence(self, scores, batch_uidx, sequence):
        """Parses a single sequence to extract user ID, recommended item, and the decoded sequence."""
        seq = self.tokenizer.decode(sequence).split(" ")

        uid_token = seq[1]
        recommended_token = seq[-1]

        if (
            not (
                uid_token.startswith(PathLanguageModelingTokenType.USER.token)
                and recommended_token.startswith(PathLanguageModelingTokenType.ITEM.token)
            )
            or recommended_token == self.tokenizer.pad_token
        ):
            return

        uid = int(uid_token[1:])
        recommended_item = int(recommended_token[1:])

        if torch.isfinite(scores[batch_uidx, recommended_item]) or recommended_item in self.used_ids[uid]:
            return

        return uid, recommended_item, seq


class SequencePostProcessorLP:
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


class CumulativeSequenceScorePostProcessor(BaseSequencePostProcessor):
    """
    Post-processor that uses the cumulative sequence score of the final
    `max_new_tokens` predicted tokens to rank sequences.
    """

    def calculate_sequence_scores(self, normalized_tuple, sequences, max_new_tokens=24):
        new_sequence_tokens = sequences[:, -max_new_tokens - 1 : -1]
        sequence_scores = []
        # Iterate over each tensor in the normalized tuple
        for i in range(max_new_tokens):
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

    def get_sequences(self, generation_outputs, max_new_tokens=24):
        user_num = generation_outputs["sequences"][:, 1].unique().numel()

        normalized_scores = self.normalize_tuple(generation_outputs["scores"])
        normalized_sequences_scores = self.calculate_sequence_scores(
            normalized_scores, generation_outputs["sequences"], max_new_tokens=max_new_tokens
        )

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        valid_sequences_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))  # false if finite
        normalized_sequences_scores = torch.where(valid_sequences_mask, -torch.inf, normalized_sequences_scores)

        sorted_indices = normalized_sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_sequences_scores = normalized_sequences_scores[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]

        return self.parse_sequences(sorted_batch_user_index, sorted_sequences, sorted_sequences_scores)


class SampleSearchSequenceScorePostProcessor(BaseSequencePostProcessor):
    """
    Post-processor that uses the sequence score of the beam search to rank sequences.

    To use only if do_sample = True and if topk and topp are set.
    """

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

    def get_sequences(self, generation_outputs, max_new_tokens=24):
        user_num = generation_outputs["sequences"][:, 1].unique().numel()

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        sequences_score = self.get_scores(sequences[:, -max_new_tokens:], generation_outputs["scores"])
        return self.parse_sequences(batch_user_index, sequences, sequences_score)


class BeamSearchSequenceScorePostProcessor(BaseSequencePostProcessor):
    """
    Post-processor that uses the sequence score of the beam search to rank sequences.
    """

    def get_sequences(self, generation_outputs, max_new_tokens=24):
        user_num = generation_outputs["sequences"][:, 1].unique().numel()

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        sorted_indices = generation_outputs["sequences_scores"].argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]
        sorted_sequences_scores = generation_outputs["sequences_scores"][sorted_indices]

        return self.parse_sequences(sorted_batch_user_index, sorted_sequences, sorted_sequences_scores)
