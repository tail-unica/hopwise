hopwise.model.sequential_recommender.sasrecf
============================================

.. py:module:: hopwise.model.sequential_recommender.sasrecf

.. autoapi-nested-parse::

   SASRecF
   ################################################



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.sasrecf.SASRecF


Module Contents
---------------

.. py:class:: SASRecF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is an extension of SASRec, which concatenates item representations and item attribute representations
   as the input to the model.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: selected_features


   .. py:attribute:: pooling_mode


   .. py:attribute:: device


   .. py:attribute:: num_feature_field


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: feature_embed_layer


   .. py:attribute:: trm_encoder


   .. py:attribute:: concat_layer


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: other_parameter_name
      :value: ['feature_embed_layer']



   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: forward(item_seq, item_seq_len)


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



