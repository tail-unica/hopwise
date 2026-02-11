hopwise.model.logits_processor
==============================

.. py:module:: hopwise.model.logits_processor

.. autoapi-nested-parse::

   hopwise.model.logits_processor
   #############################
   Common logits processor in recommender system



Classes
-------

.. autoapisummary::

   hopwise.model.logits_processor.LogitsProcessor
   hopwise.model.logits_processor.LogitsProcessorList
   hopwise.model.logits_processor.ConstrainedLogitsProcessorWordLevel
   hopwise.model.logits_processor.PrefixConstrainedLogitsProcessorWordLevel
   hopwise.model.logits_processor.PLMLogitsProcessorWordLevel


Module Contents
---------------

.. py:class:: LogitsProcessor

   Abstract base class for all logit processors that can be applied during generation.
   Copy of HuggingFace's LogitsProcessor.


   .. py:method:: __call__(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor
      :abstractmethod:



.. py:class:: LogitsProcessorList

   Bases: :py:obj:`list`


   This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
   This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
   inputs.
   Copy of HuggingFace's LogitsProcessorList.


   .. py:method:: __call__(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor

      :param input_ids: Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
      :type input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
      :param scores: Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                     beam search or log softmax for each vocabulary token when using beam search
      :type scores: `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`
      :param kwargs: Additional kwargs that are specific to a logits processor.
      :type kwargs: `Dict[str, Any]`, *optional*

      :returns:     The processed prediction scores.
      :rtype: `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`



.. py:class:: ConstrainedLogitsProcessorWordLevel(tokenized_ckg, tokenized_used_ids, max_sequence_length, tokenizer, mask_cache_size=3 * 10**4, pos_candidates_cache_size=1 * 10**5, task=KnowledgeEvaluationType.REC, **kwargs)

   Bases: :py:obj:`LogitsProcessor`


   Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
   this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
   If task is link prediction (LP) logit processor forces last token to reachable ones


   .. py:attribute:: tokenized_ckg


   .. py:attribute:: tokenized_used_ids


   .. py:attribute:: max_sequence_length


   .. py:attribute:: tokenizer


   .. py:attribute:: bos_token_id


   .. py:attribute:: pos_candidates_cache


   .. py:attribute:: mask_cache


   .. py:attribute:: task


   .. py:method:: is_bos_token_in_input(input_ids)

      Check if the input contains a BOS token. Checking the first sequence is enough.



   .. py:method:: __call__(input_ids, scores)


   .. py:method:: process_scores_rec(input_ids, idx)

      Process each score based on input length and update mask list.



   .. py:method:: process_scores_lp(input_ids, idx)

      Process each score based on input length or skip.



   .. py:method:: is_next_token_entity(input_ids)


   .. py:method:: get_current_key(input_ids, idx)


   .. py:method:: get_candidates_rec(key1, key2=None)

      :param key1:
      :param key2: if key2 is not None, it returns entity candidates, otherwise relation candidates



   .. py:method:: get_candidates_lp(key)


   .. py:method:: get_banned_mask(key, candidate_tokens)

      Retrieve or cache the banned token mask for a specific key.



.. py:class:: PrefixConstrainedLogitsProcessorWordLevel(tokenized_ckg, tokenized_used_ids, max_sequence_length, tokenizer, **kwargs)

   Bases: :py:obj:`ConstrainedLogitsProcessorWordLevel`


   Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
   this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
   If task is link prediction (LP) logit processor forces last token to reachable ones


   .. py:attribute:: mask_cache
      :value: None



   .. py:method:: __call__(input_ids, scores)


.. py:class:: PLMLogitsProcessorWordLevel(tokenized_ckg, tokenized_used_ids, max_sequence_length, tokenizer, pos_candidates_cache_size=1 * 10**5, task=KnowledgeEvaluationType.REC, **kwargs)

   Bases: :py:obj:`LogitsProcessor`


   https://dl.acm.org/doi/pdf/10.1145/3485447.3511937
   Constraint decoding strategy for PLM, it forces the model to generate alternatively entities and relations


   .. py:attribute:: tokenized_ckg


   .. py:attribute:: tokenized_used_ids


   .. py:attribute:: max_sequence_length


   .. py:attribute:: tokenizer


   .. py:attribute:: bos_token_id


   .. py:attribute:: pos_candidates_cache


   .. py:attribute:: task


   .. py:attribute:: entity_token_ids


   .. py:attribute:: relation_token_ids


   .. py:method:: __call__(input_ids, scores)


   .. py:method:: is_bos_token_in_input(input_ids)

      Check if the input contains a BOS token. Checking the first sequence is enough.



   .. py:method:: is_next_token_entity(input_ids)


   .. py:method:: process_scores(input_ids, idx)

      Process each score based on input length and update mask to allow only entities or only relations.



