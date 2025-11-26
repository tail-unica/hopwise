# @Time   : 2025
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.model.logits_processor
#############################
Common logits processor in recommender system
"""

import inspect

import numpy as np
import torch
from cachetools import LFUCache

from hopwise.utils import KnowledgeEvaluationType


class LogitsProcessor:
    """
    Abstract base class for all logit processors that can be applied during generation.
    Copy of HuggingFace's LogitsProcessor.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    Copy of HuggingFace's LogitsProcessorList.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:  # noqa: PLR2004
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


class ConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    """
    Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
    this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
    If task is link prediction (LP) logit processor forces last token to reachable ones
    """

    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenized_ckg = tokenized_ckg
        self.tokenized_used_ids = tokenized_used_ids
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.pos_candidates_cache = LFUCache(pos_candidates_cache_size)
        self.mask_cache = LFUCache(mask_cache_size)
        self.task = task

        if self.task == KnowledgeEvaluationType.LP:
            self.special_tokens_ids = [
                self.tokenizer.encode(x, add_special_tokens=False)[0]
                for x in self.tokenizer.all_special_tokens_extended
            ]
        else:
            self.special_tokens_ids = None

    def is_bos_token_in_input(self, input_ids):
        """Check if the input contains a BOS token. Checking the first sequence is enough."""
        return (input_ids[0, 0] == self.bos_token_id).item()

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            # Determine whether the next token to generate is a relation or an entity:
            # - relation: only the last entity is needed (1 token) → for [user123] last_n_tokens = 1
            # - entity: the last 2 tokens are needed (entity, relation) → for [user123, watched] last_n_tokens = 2
            # Apply deduplication: select unique sequences based only on the relevant last tokens (1 or 2)
            # This avoids recomputing the same mask for sequences that share the same context
            last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
            _, input_ids_indices, input_ids_inv = np.unique(
                input_ids.cpu().numpy()[:, -last_n_tokens:], axis=0, return_index=True, return_inverse=True
            )
            unique_input_ids = input_ids[input_ids_indices]

        full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
        for idx in range(unique_input_ids.shape[0]):
            if self.task == KnowledgeEvaluationType.REC:
                key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
            elif self.task == KnowledgeEvaluationType.LP:
                key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)

            banned_mask = self.get_banned_mask(key, candidate_tokens)

            if banned_mask.all():
                banned_mask[self.tokenizer.pad_token_id] = False

            full_mask[idx] = banned_mask

        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            scores[full_mask] = -torch.inf

        return scores

    def process_scores_rec(self, input_ids, idx):
        """Process each score based on input length and update mask list."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        key = self.get_current_key(input_ids, idx)
        if current_len == self.max_sequence_length - 1 - has_bos_token:
            current_uid = input_ids[idx, int(has_bos_token)].item()
            uid_cond_key = (current_uid, *key)

            candidate_tokens = self.pos_candidates_cache.get(uid_cond_key)
            if candidate_tokens is None:
                candidate_tokens = self.get_candidates_rec(*key)

                # Get user positives
                user_used_ids = self.tokenized_used_ids[current_uid]
                # Select negatives
                candidate_tokens = list(candidate_tokens - user_used_ids)
                # Useless if during evaluation a user is seen once
                self.pos_candidates_cache[uid_cond_key] = candidate_tokens
        else:
            candidate_tokens = list(self.get_candidates_rec(*key))

        return key, candidate_tokens

    def process_scores_lp(self, input_ids, idx):
        """Process each score based on input length or skip."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        key, candidate_tokens = None, None
        if current_len % 2 == has_bos_token:
            key = self.get_current_key(input_ids, idx)
            candidate_tokens = self.get_candidates_lp(key)

        return key, candidate_tokens

    def is_next_token_entity(self, input_ids):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        # bos_token determines if the current length is even or odd
        return current_len % 2 == has_bos_token

    def get_current_key(self, input_ids, idx):
        if self.is_next_token_entity(input_ids):
            return input_ids[idx, -2].item(), input_ids[idx, -1].item()
        else:
            # The next token is a relation
            return (input_ids[idx, -1].item(),)

    def get_candidates_rec(self, key1, key2=None):
        """
        :param key1:
        :param key2: if key2 is not None, it returns entity candidates, otherwise relation candidates
        """
        if key1 in self.tokenized_ckg:
            if key2 is not None and key2 in self.tokenized_ckg[key1]:
                # return tail given head + relation
                return self.tokenized_ckg[key1][key2]
            else:
                # return relations given head
                return set(self.tokenized_ckg[key1].keys())
        else:
            raise ValueError(f"Key {key1} ('{self.tokenizer.convert_ids_to_tokens(key1)}') not found in tokenized_ckg")

    def get_candidates_lp(self, key):
        return list(self.tokenized_used_ids[key]) + self.special_tokens_ids

    def get_banned_mask(self, key, candidate_tokens):
        """Retrieve or cache the banned token mask for a specific key."""
        banned_mask = self.mask_cache.get(key)
        if banned_mask is None:
            banned_mask = np.ones(len(self.tokenizer), dtype=bool)
            banned_mask[candidate_tokens] = False
            self.mask_cache[key] = banned_mask
        return banned_mask


class PrefixConstrainedLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        **kwargs,
    ):
        super().__init__(
            tokenized_ckg,
            tokenized_used_ids,
            max_sequence_length,
            tokenizer,
            **kwargs,
        )
        self.mask_cache = None

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        if current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            indices = []
            masked_scores = torch.full_like(scores, -torch.inf)
            for idx in range(scores.shape[0]):
                _, candidate_tokens = self.process_scores(input_ids, idx, current_len)

                candidate_tokens = torch.LongTensor(candidate_tokens, device=scores.device)
                indices.append(candidate_tokens)
                masked_scores[idx].scatter_(dim=-1, index=candidate_tokens, src=scores[idx])
            scores = masked_scores

        return scores


class PLMLogitsProcessorWordLevel(LogitsProcessor):
    """
    https://dl.acm.org/doi/pdf/10.1145/3485447.3511937
    Constraint decoding strategy for PLM, it forces the model to generate alternatively entities and relations
    """

    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenized_ckg = tokenized_ckg
        self.tokenized_used_ids = tokenized_used_ids
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.pos_candidates_cache = LFUCache(pos_candidates_cache_size)
        self.task = task

        if self.task == KnowledgeEvaluationType.LP:
            self.special_tokens_ids = [
                self.tokenizer.encode(x, add_special_tokens=False)[0]
                for x in self.tokenizer.all_special_tokens_extended
            ]
        else:
            self.special_tokens_ids = None

        self.entity_token_ids = torch.LongTensor(list(set(self.tokenized_ckg.keys())))
        self.relation_token_ids = torch.LongTensor(
            list(set([rel for rel_dict in self.tokenized_ckg.values() for rel in rel_dict.keys()]))
        )

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        if self.task == KnowledgeEvaluationType.REC and current_len == (self.max_sequence_length - 1 - has_bos_token):
            user_idx = int(has_bos_token)
            _, input_ids_indices, input_ids_inv = np.unique(
                input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
            )
            unique_input_ids = input_ids[input_ids_indices]

            full_mask = np.ones((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
            for idx in range(unique_input_ids.shape[0]):
                candidate_tokens = self.process_scores(unique_input_ids, idx)
                full_mask[idx, candidate_tokens] = False

            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            # Paths are expected to be the same type and length, so we can use the same mask for all
            full_mask = np.ones((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
            candidate_tokens = self.process_scores(input_ids, 0)
            full_mask[:, candidate_tokens] = False
            scores[full_mask] = -torch.inf

        return scores

    def is_bos_token_in_input(self, input_ids):
        """Check if the input contains a BOS token. Checking the first sequence is enough."""
        return (input_ids[0, 0] == self.bos_token_id).item()

    def is_next_token_entity(self, input_ids):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        # bos_token determines if the current length is even or odd
        return current_len % 2 == has_bos_token

    def process_scores(self, input_ids, idx):
        """Process each score based on input length and update mask to allow only entities or only relations."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if current_len == self.max_sequence_length - 1 - has_bos_token:
            current_uid = input_ids[idx, int(has_bos_token)].item()
            candidate_tokens = self.pos_candidates_cache.get(current_uid)
            if candidate_tokens is None:
                candidate_tokens = np.arange(len(self.tokenizer))

                user_used_ids = self.tokenized_used_ids[current_uid]
                candidate_tokens = np.setdiff1d(candidate_tokens, list(user_used_ids), assume_unique=True)
                self.pos_candidates_cache[current_uid] = candidate_tokens
        elif self.is_next_token_entity(input_ids):
            candidate_tokens = self.entity_token_ids
        else:
            candidate_tokens = self.relation_token_ids

        return candidate_tokens
