hopwise.model.sequential_recommender.lightsans
==============================================

.. py:module:: hopwise.model.sequential_recommender.lightsans

.. autoapi-nested-parse::

   LightSANs
   ################################################
   Reference:
       Xin-Yan Fan et al. "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." in SIGIR 2021.
   Reference:
       https://github.com/BELIEVEfxy/LightSANs



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.lightsans.LightSANs


Module Contents
---------------

.. py:class:: LightSANs(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: k_interests


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: seq_len


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: embedding_layer(item_seq)


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



