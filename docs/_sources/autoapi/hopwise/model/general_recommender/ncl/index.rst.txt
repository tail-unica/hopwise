hopwise.model.general_recommender.ncl
=====================================

.. py:module:: hopwise.model.general_recommender.ncl

.. autoapi-nested-parse::

   NCL
   ################################################

   Reference:
       Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.ncl.NCL


Module Contents
---------------

.. py:class:: NCL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NCL is a neighborhood-enriched contrastive learning paradigm for graph collaborative filtering.
   Both structural and semantic neighbors are explicitly captured as contrastive learning objects.


   .. py:attribute:: input_type


   .. py:attribute:: latent_dim


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: ssl_temp


   .. py:attribute:: ssl_reg


   .. py:attribute:: hyper_layers


   .. py:attribute:: alpha


   .. py:attribute:: proto_reg


   .. py:attribute:: k


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



   .. py:attribute:: user_centroids
      :value: None



   .. py:attribute:: user_2cluster
      :value: None



   .. py:attribute:: item_centroids
      :value: None



   .. py:attribute:: item_2cluster
      :value: None



   .. py:method:: e_step()


   .. py:method:: run_kmeans(x)

      Run K-means algorithm to get k clusters of the input tensor x



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]



   .. py:method:: forward()


   .. py:method:: ProtoNCE_loss(node_embedding, user, item)


   .. py:method:: ssl_layer_loss(current_embedding, previous_embedding, user, item)


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



