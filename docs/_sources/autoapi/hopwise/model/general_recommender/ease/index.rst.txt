hopwise.model.general_recommender.ease
======================================

.. py:module:: hopwise.model.general_recommender.ease

.. autoapi-nested-parse::

   EASE
   ################################################
   Reference:
       Harald Steck. "Embarrassingly Shallow Autoencoders for Sparse Data" in WWW 2019.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.ease.EASE


Module Contents
---------------

.. py:class:: EASE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   EASE is a linear model for collaborative filtering, which combines the
   strengths of auto-encoders and neighborhood-based approaches.



   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: dummy_param


   .. py:attribute:: item_similarity


   .. py:attribute:: interaction_matrix


   .. py:attribute:: other_parameter_name
      :value: ['interaction_matrix', 'item_similarity']



   .. py:attribute:: device


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



