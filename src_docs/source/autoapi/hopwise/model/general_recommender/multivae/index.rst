hopwise.model.general_recommender.multivae
==========================================

.. py:module:: hopwise.model.general_recommender.multivae

.. autoapi-nested-parse::

   MultiVAE
   ################################################
   Reference:
       Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering." in WWW 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.multivae.MultiVAE


Module Contents
---------------

.. py:class:: MultiVAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   MultiVAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

   We implement the MultiVAE model with only user dataloader.


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


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix)


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



