hopwise.model.sequence_postprocessor
====================================

.. py:module:: hopwise.model.sequence_postprocessor

.. autoapi-nested-parse::

   hopwise.model.postprocessor
   #######################
   Common post-processors for path sequences in path language modeling recommender systems.



Classes
-------

.. autoapisummary::

   hopwise.model.sequence_postprocessor.BaseSequencePostProcessor
   hopwise.model.sequence_postprocessor.SequencePostProcessorLP
   hopwise.model.sequence_postprocessor.CumulativeSequenceScorePostProcessor
   hopwise.model.sequence_postprocessor.SampleSearchSequenceScorePostProcessor
   hopwise.model.sequence_postprocessor.BeamSearchSequenceScorePostProcessor


Module Contents
---------------

.. py:class:: BaseSequencePostProcessor(tokenizer, used_ids, item_num, topk=10)

   Base class for sequence score post-processors.


   .. py:attribute:: tokenizer


   .. py:attribute:: used_ids


   .. py:attribute:: item_num


   .. py:attribute:: topk
      :value: 10



   .. py:method:: get_sequences(generation_outputs, max_new_tokens=24)
      :abstractmethod:


      This method should be implemented by subclasses to extract sequences and their scores.

      :param generation_outputs: A mapping containing the generated sequences and their scores.
      :param max_new_tokens: The maximum number of new tokens to consider for scoring.



   .. py:method:: parse_sequences(user_index, sequences, sequences_scores)

      Parses the sequences and scores to extract user IDs, recommended items, and their scores.

      :param user_index: A tensor containing user indices.
      :type user_index: torch.Tensor
      :param sequences: A tensor containing the generated sequences.
      :type sequences: torch.Tensor
      :param sequences_scores: A tensor containing the scores for each sequence.
      :type sequences_scores: torch.Tensor

      :returns:

                A tuple containing:

                    - scores (torch.Tensor): A tensor of shape (user_num, item_num) containing the scores.
                    - user_topk_sequences (list): A list of lists with
                        [user_id, recommended_item, score, decoded_sequence].
      :rtype: tuple



   .. py:method:: _parse_single_sequence(scores, batch_uidx, sequence)

      Parses a single sequence to extract user ID, recommended item, and the decoded sequence.



.. py:class:: SequencePostProcessorLP(tokenizer, kg_positives, K=10, max_new_tokens=24)

   .. py:attribute:: tokenizer


   .. py:attribute:: kg_positives


   .. py:attribute:: topk


   .. py:attribute:: topk_sequences


   .. py:attribute:: max_new_tokens
      :value: 24



   .. py:attribute:: K
      :value: 10



   .. py:method:: update_topk(generate_outputs)


   .. py:method:: reset_topks()


.. py:class:: CumulativeSequenceScorePostProcessor(tokenizer, used_ids, item_num, topk=10)

   Bases: :py:obj:`BaseSequencePostProcessor`


   Post-processor that uses the cumulative sequence score of the final
   `max_new_tokens` predicted tokens to rank sequences.


   .. py:method:: calculate_sequence_scores(normalized_tuple, sequences, max_new_tokens=24)


   .. py:method:: normalize_tuple(logits_tuple)


   .. py:method:: get_sequences(generation_outputs, max_new_tokens=24)

      This method should be implemented by subclasses to extract sequences and their scores.

      :param generation_outputs: A mapping containing the generated sequences and their scores.
      :param max_new_tokens: The maximum number of new tokens to consider for scoring.



.. py:class:: SampleSearchSequenceScorePostProcessor(tokenizer, used_ids, item_num, topk=10)

   Bases: :py:obj:`BaseSequencePostProcessor`


   Post-processor that uses the sequence score of the beam search to rank sequences.

   To use only if do_sample = True and if topk and topp are set.


   .. py:method:: get_scores(sequences, scores)


   .. py:method:: get_sequences(generation_outputs, max_new_tokens=24)

      This method should be implemented by subclasses to extract sequences and their scores.

      :param generation_outputs: A mapping containing the generated sequences and their scores.
      :param max_new_tokens: The maximum number of new tokens to consider for scoring.



.. py:class:: BeamSearchSequenceScorePostProcessor(tokenizer, used_ids, item_num, topk=10)

   Bases: :py:obj:`BaseSequencePostProcessor`


   Post-processor that uses the sequence score of the beam search to rank sequences.


   .. py:method:: get_sequences(generation_outputs, max_new_tokens=24)

      This method should be implemented by subclasses to extract sequences and their scores.

      :param generation_outputs: A mapping containing the generated sequences and their scores.
      :param max_new_tokens: The maximum number of new tokens to consider for scoring.



