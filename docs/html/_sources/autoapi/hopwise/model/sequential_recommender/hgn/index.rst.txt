hopwise.model.sequential_recommender.hgn
========================================

.. py:module:: hopwise.model.sequential_recommender.hgn

.. autoapi-nested-parse::

   HGN
   ################################################

   Reference:
       Chen Ma et al. "Hierarchical Gating Networks for Sequential Recommendation."in SIGKDD 2019




Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.hgn.HGN


Module Contents
---------------

.. py:class:: HGN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   HGN sets feature gating and instance gating to get the important feature and item for predicting the next item


   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: pool_type


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: w1


   .. py:attribute:: w2


   .. py:attribute:: b


   .. py:attribute:: w3


   .. py:attribute:: w4


   .. py:attribute:: item_embedding_for_prediction


   .. py:attribute:: sigmoid


   .. py:attribute:: loss_type


   .. py:method:: reg_loss(user_embedding, item_embedding, seq_item_embedding)


   .. py:method:: _init_weights(module)


   .. py:method:: feature_gating(seq_item_embedding, user_embedding)

      Choose the features that will be sent to the next stage(more important feature, more focus)



   .. py:method:: instance_gating(user_item, user_embedding)

      Choose the last click items that will influence the prediction( more important more chance to get attention)



   .. py:method:: forward(seq_item, user)


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



