from hopwise.model.layers import ConstrainedLogitsProcessorWordLevel


class PLMLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    """
    https://dl.acm.org/doi/pdf/10.1145/3485447.3511937
    Constraint decoding strategy for PLM, it forces the model to generate alternatively entities and relations
    """

    def __init__(
        self,
        tokenized_kg,
        force_token_map,
        total_length,
        tokenizer,
        num_return_sequences,
        id_to_uid_token_map,
        eos_token_ids,
        ent_mask,
        rel_mask,
        token_id_to_token,
        mask_cache_size=3 * 10**4,
        cand_cache_size=1 * 10**5,
        **kwargs,
    ):
        super().__init__(
            tokenized_kg,
            force_token_map,
            total_length,
            tokenizer,
            num_return_sequences,
            id_to_uid_token_map,
            eos_token_ids,
            mask_cache_size=mask_cache_size,
            cand_cache_size=cand_cache_size,
            **kwargs,
        )
        self.ent_ids = [idx for idx, elem in enumerate(ent_mask) if elem > 0]
        self.rel_ids = [idx for idx, elem in enumerate(rel_mask) if elem > 0]

        self.ent_mask = [elem > 0 for idx, elem in enumerate(ent_mask)]
        self.rel_mask = [elem > 0 for idx, elem in enumerate(rel_mask)]

        self.token_id_to_token = token_id_to_token

    @staticmethod
    def get_current_key(input_ids, idx, current_len):
        return int(current_len % 2 == 1)

    def process_scores(self, input_ids, idx, current_len):
        """Process each score based on input length and update mask list."""
        key = self.get_current_key(input_ids, idx, current_len)
        current_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
        return self.get_key_and_candidates(key, idx, current_len, current_uid)

    def get_key_and_candidates(self, key, idx, current_len, current_uid):
        """Retrieve candidate tokens and update key based on current length."""
        if current_len % 2 == 1:
            candidate_tokens = self.ent_ids

            if current_len == self.max_sequence_length - 1:
                candidate_tokens = self.force_token_map[current_uid]
                key = current_uid, idx
        else:
            candidate_tokens = self.rel_ids
            key = 0

        return key, candidate_tokens
