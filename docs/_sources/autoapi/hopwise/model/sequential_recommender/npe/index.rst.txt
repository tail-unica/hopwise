hopwise.model.sequential_recommender.npe
========================================

.. py:module:: hopwise.model.sequential_recommender.npe

.. autoapi-nested-parse::

   NPE
   ################################################

   Reference:
       ThaiBinh Nguyen, et al. "NPE: Neural Personalized Embedding for Collaborative Filtering" in IJCAI 2018.

   Reference code:
       https://github.com/wubinzzu/NeuRec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.npe.NPE


Module Contents
---------------

.. py:class:: NPE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   models a user’s click to an item in two terms: the personal preference of the user for the item,
   and the relationships between this item and other items clicked by the user



   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: embedding_seq_item


   .. py:attribute:: relu


   .. py:attribute:: dropout


   .. py:attribute:: loss_type


   .. py:method:: _init_weights(module)


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



