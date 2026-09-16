hopwise.model.sequential_recommender.fossil
===========================================

.. py:module:: hopwise.model.sequential_recommender.fossil

.. autoapi-nested-parse::

   FOSSIL
   ################################################

   Reference:
       Ruining He et al. "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation." in ICDM 2016.




Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.fossil.FOSSIL


Module Contents
---------------

.. py:class:: FOSSIL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   FOSSIL uses similarity of the items as main purpose and uses high MC as a way of sequential preference improve of
   ability of sequential recommendation



   .. py:attribute:: n_users


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: order_len


   .. py:attribute:: reg_weight


   .. py:attribute:: alpha


   .. py:attribute:: item_embedding


   .. py:attribute:: user_lambda


   .. py:attribute:: lambda_


   .. py:attribute:: loss_type


   .. py:method:: inverse_seq_item_embedding(seq_item_embedding, seq_item_len)

      Inverse seq_item_embedding like this (simple to 2-dim):

      [1,2,3,0,0,0] -- ??? -- >> [0,0,0,1,2,3]

      first: [0,0,0,0,0,0] concat [1,2,3,0,0,0]

      using gather_indexes: to get one by one

      first get 3,then 2,last 1



   .. py:method:: reg_loss(user_embedding, item_embedding, seq_output)


   .. py:method:: init_weights(module)


   .. py:method:: forward(seq_item, seq_item_len, user)


   .. py:method:: get_high_order_Markov(high_order_item_embedding, user)

      In order to get the inference of past items and the user's taste to the current predict item



   .. py:method:: get_similarity(seq_item_embedding, seq_item_len)

      In order to get the inference of past items to the current predict item



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



