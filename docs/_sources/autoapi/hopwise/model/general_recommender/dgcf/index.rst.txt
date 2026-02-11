hopwise.model.general_recommender.dgcf
======================================

.. py:module:: hopwise.model.general_recommender.dgcf

.. autoapi-nested-parse::

   DGCF
   ################################################
   Reference:
       Wang Xiang et al. "Disentangled Graph Collaborative Filtering." in SIGIR 2020.

   Reference code:
       https://github.com/xiangwang1223/disentangled_graph_collaborative_filtering



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.dgcf.DGCF


Functions
---------

.. autoapisummary::

   hopwise.model.general_recommender.dgcf.sample_cor_samples


Module Contents
---------------

.. py:function:: sample_cor_samples(n_users, n_items, cor_batch_size)

   This is a function that sample item ids and user ids.

   :param n_users: number of users in total
   :type n_users: int
   :param n_items: number of items in total
   :type n_items: int
   :param cor_batch_size: number of id to sample
   :type cor_batch_size: int

   :returns: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.
   :rtype: list

   .. note::

      We have to sample some embedded representations out of all nodes.
      Because we have no way to store cor-distance for each pair.


.. py:class:: DGCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   DGCF is a disentangled representation enhanced matrix factorization model.
   The interaction matrix of :math:`n_{users} \times n_{items}` is decomposed to :math:`n_{factors}` intent graph,
   we carefully design the data interface and use sparse tensor to train and test efficiently.
   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: interaction_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: n_factors


   .. py:attribute:: n_iterations


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: cor_weight


   .. py:attribute:: cor_batch_size


   .. py:attribute:: all_h_list


   .. py:attribute:: all_t_list


   .. py:attribute:: edge2head


   .. py:attribute:: head2edge


   .. py:attribute:: tail2edge


   .. py:attribute:: edge2head_mat


   .. py:attribute:: head2edge_mat


   .. py:attribute:: tail2edge_mat


   .. py:attribute:: num_edge


   .. py:attribute:: num_node


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: softmax


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: _build_sparse_tensor(indices, values, size)


   .. py:method:: _get_ego_embeddings()


   .. py:method:: build_matrix(A_values)

      Get the normalized interaction matrix of users and items according to A_values.

      Construct the square matrix from the training data and normalize it
      using the laplace matrix.

      :param A_values: (num_edge, n_factors)
      :type A_values: torch.cuda.FloatTensor

      .. math::
          A_{hat} = D^{-0.5} \times A \times D^{-0.5}

      :returns: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
      :rtype: torch.cuda.FloatTensor



   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: create_cor_loss(cor_u_embeddings, cor_i_embeddings)

      Calculate the correlation loss for a sampled users and items.

      :param cor_u_embeddings: (cor_batch_size, n_factors)
      :type cor_u_embeddings: torch.cuda.FloatTensor
      :param cor_i_embeddings: (cor_batch_size, n_factors)
      :type cor_i_embeddings: torch.cuda.FloatTensor

      :returns: correlation loss.
      :rtype: torch.Tensor



   .. py:method:: _create_distance_correlation(X1, X2)


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



