hopwise.model.general_recommender.random
========================================

.. py:module:: hopwise.model.general_recommender.random

.. autoapi-nested-parse::

   Random
   ################################################



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.random.Random


Module Contents
---------------

.. py:class:: Random(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   Random is an fundamental model that recommends random items.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: fake_loss


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



