hopwise.model.sequential_recommender.bert4rec
=============================================

.. py:module:: hopwise.model.sequential_recommender.bert4rec

.. autoapi-nested-parse::

   BERT4Rec
   ################################################

   Reference:
       Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
       In CIKM 2019.

   Reference code:
       The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.bert4rec.BERT4Rec


Module Contents
---------------

.. py:class:: BERT4Rec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: mask_ratio


   .. py:attribute:: MASK_ITEM_SEQ


   .. py:attribute:: POS_ITEMS


   .. py:attribute:: NEG_ITEMS


   .. py:attribute:: MASK_INDEX


   .. py:attribute:: loss_type


   .. py:attribute:: initializer_range


   .. py:attribute:: mask_token


   .. py:attribute:: mask_item_length


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: output_ffn


   .. py:attribute:: output_gelu


   .. py:attribute:: output_ln


   .. py:attribute:: output_bias


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: reconstruct_test_data(item_seq, item_seq_len)

      Add mask token at the last position according to the lengths of item_seq



   .. py:method:: forward(item_seq)


   .. py:method:: multi_hot_embed(masked_index, max_length)

      For memory, we only need calculate loss for masked position.
      Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
      gathering the masked position hidden representation.

      .. rubric:: Examples

      sequence: [1 2 3 4 5]

      masked_sequence: [1 mask 3 mask 5]

      masked_index: [1, 3]

      max_length: 5

      multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]



   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



