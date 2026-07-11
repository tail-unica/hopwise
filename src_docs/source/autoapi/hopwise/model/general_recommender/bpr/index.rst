hopwise.model.general_recommender.bpr
=====================================

.. py:module:: hopwise.model.general_recommender.bpr

.. autoapi-nested-parse::

   BPR
   ################################################
   Reference:
       Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.bpr.BPR


Module Contents
---------------

.. py:class:: BPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   BPR is a basic matrix factorization model that be trained in the pairwise way.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: loss


   .. py:method:: get_user_embedding(user)

      Get a batch of user embedding tensor according to input user's id.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor

      :returns: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: get_item_embedding(item)

      Get a batch of item embedding tensor according to input item's id.

      :param item: The input tensor that contains item's id, shape: [batch_size, ]
      :type item: torch.LongTensor

      :returns: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: forward(user, item)


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



