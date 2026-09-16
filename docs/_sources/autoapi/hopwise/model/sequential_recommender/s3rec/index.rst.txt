hopwise.model.sequential_recommender.s3rec
==========================================

.. py:module:: hopwise.model.sequential_recommender.s3rec

.. autoapi-nested-parse::

   S3Rec
   ################################################

   Reference:
       Kun Zhou and Hui Wang et al. "S^3-Rec: Self-Supervised Learning
       for Sequential Recommendation with Mutual Information Maximization"
       In CIKM 2020.

   Reference code:
       https://github.com/RUCAIBox/CIKM2020-S3Rec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.s3rec.S3Rec


Module Contents
---------------

.. py:class:: S3Rec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   S3Rec is the first work to incorporate self-supervised learning in
   sequential recommendation.

   .. note::

      Under this framework, we need reconstruct the pretraining data,
      which would affect the pre-training speed.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: FEATURE_FIELD


   .. py:attribute:: FEATURE_LIST


   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


   .. py:attribute:: mask_ratio


   .. py:attribute:: aap_weight


   .. py:attribute:: mip_weight


   .. py:attribute:: map_weight


   .. py:attribute:: sp_weight


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: n_items


   .. py:attribute:: mask_token


   .. py:attribute:: n_features


   .. py:attribute:: item_feat


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: feature_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: aap_norm


   .. py:attribute:: mip_norm


   .. py:attribute:: map_norm


   .. py:attribute:: sp_norm


   .. py:attribute:: loss_fct


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _associated_attribute_prediction(sequence_output, feature_embedding)


   .. py:method:: _masked_item_prediction(sequence_output, target_item_emb)


   .. py:method:: _masked_attribute_prediction(sequence_output, feature_embedding)


   .. py:method:: _segment_prediction(context, segment_emb)


   .. py:method:: forward(item_seq, bidirectional=True)


   .. py:method:: pretrain(features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment)

      Pretrain out model using four pre-training tasks:

      1. Associated Attribute Prediction

      2. Masked Item Prediction

      3. Masked Attribute Prediction

      4. Segment Prediction



   .. py:method:: _neg_sample(item_set)


   .. py:method:: _padding_zero_at_left(sequence)


   .. py:method:: reconstruct_pretrain_data(item_seq, item_seq_len)

      Generate pre-training data for the pre-training stage.



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



