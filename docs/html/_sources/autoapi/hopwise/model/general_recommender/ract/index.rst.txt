hopwise.model.general_recommender.ract
======================================

.. py:module:: hopwise.model.general_recommender.ract

.. autoapi-nested-parse::

   RaCT
   ################################################
   Reference:
       Sam Lobel et al. "RaCT: Towards Amortized Ranking-Critical Training for Collaborative Filtering." in ICLR 2020.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.ract.RaCT


Module Contents
---------------

.. py:class:: RaCT(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   RaCT is a collaborative filtering model which uses methods based on actor-critic reinforcement learning for training.

   We implement the RaCT model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: layers


   .. py:attribute:: lat_dim


   .. py:attribute:: drop_out


   .. py:attribute:: anneal_cap


   .. py:attribute:: total_anneal_steps


   .. py:attribute:: update
      :value: 0



   .. py:attribute:: encode_layer_dims


   .. py:attribute:: decode_layer_dims


   .. py:attribute:: encoder


   .. py:attribute:: decoder


   .. py:attribute:: critic_layers


   .. py:attribute:: metrics_k


   .. py:attribute:: number_of_seen_items
      :value: 0



   .. py:attribute:: number_of_unseen_items
      :value: 0



   .. py:attribute:: critic_layer_dims


   .. py:attribute:: input_matrix
      :value: None



   .. py:attribute:: predict_matrix
      :value: None



   .. py:attribute:: true_matrix
      :value: None



   .. py:attribute:: critic_net


   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix)


   .. py:method:: calculate_actor_loss(interaction)


   .. py:method:: construct_critic_input(actor_loss)


   .. py:method:: construct_critic_layers(layer_dims)


   .. py:method:: calculate_ndcg(predict_matrix, true_matrix, input_matrix, k)


   .. py:method:: critic_forward(actor_loss)


   .. py:method:: calculate_critic_loss(interaction)


   .. py:method:: calculate_ac_loss(interaction)


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



