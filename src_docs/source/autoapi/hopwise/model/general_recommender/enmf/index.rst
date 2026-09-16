hopwise.model.general_recommender.enmf
======================================

.. py:module:: hopwise.model.general_recommender.enmf

.. autoapi-nested-parse::

   ENMF
   ################################################
   Reference:
       Chong Chen et al. "Efficient Neural Matrix Factorization without Sampling for Recommendation." in TOIS 2020.

   Reference code:
       https://github.com/chenchongthu/ENMF



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.enmf.ENMF


Module Contents
---------------

.. py:class:: ENMF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   ENMF is an efficient non-sampling model for general recommendation.
   In order to run non-sampling model, please set the neg_sampling parameter as None .



   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: reg_weight


   .. py:attribute:: negative_weight


   .. py:attribute:: history_item_matrix


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: H_i


   .. py:attribute:: dropout


   .. py:method:: reg_loss()

      Calculate the reg loss for embedding layers and mlp layers

      :returns: reg loss
      :rtype: torch.Tensor



   .. py:method:: forward(user)


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



