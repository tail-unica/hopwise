hopwise.model.general_recommender.pop
=====================================

.. py:module:: hopwise.model.general_recommender.pop

.. autoapi-nested-parse::

   Pop
   ################################################



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.pop.Pop


Module Contents
---------------

.. py:class:: Pop(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   Pop is an fundamental model that always recommend the most popular item.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: item_cnt


   .. py:attribute:: max_cnt
      :value: None



   .. py:attribute:: fake_loss


   .. py:attribute:: other_parameter_name
      :value: ['item_cnt', 'max_cnt']



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



