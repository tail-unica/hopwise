hopwise.model.general_recommender.ngcf
======================================

.. py:module:: hopwise.model.general_recommender.ngcf

.. autoapi-nested-parse::

   NGCF
   ################################################
   Reference:
       Xiang Wang et al. "Neural Graph Collaborative Filtering." in SIGIR 2019.

   Reference code:
       https://github.com/xiangwang1223/neural_graph_collaborative_filtering



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.ngcf.NGCF


Module Contents
---------------

.. py:class:: NGCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NGCF is a model that incorporate GNN for recommendation.
   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size_list


   .. py:attribute:: node_dropout


   .. py:attribute:: message_dropout


   .. py:attribute:: reg_weight


   .. py:attribute:: sparse_dropout


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: GNNlayers


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: norm_adj_matrix


   .. py:attribute:: eye_matrix


   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)



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



