hopwise.model.general_recommender.lightgcn
==========================================

.. py:module:: hopwise.model.general_recommender.lightgcn

.. autoapi-nested-parse::

   LightGCN
   ################################################

   Reference:
       Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

   Reference code:
       https://github.com/kuandeng/LightGCN



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.lightgcn.LightGCN


Module Contents
---------------

.. py:class:: LightGCN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   LightGCN is a GCN-based recommender model.

   LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
   collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
   propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
   learned at all layers as the final embedding.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: latent_dim


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: require_pow


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: norm_adj_matrix


   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]



   .. py:method:: forward()


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



