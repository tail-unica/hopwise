hopwise.model.context_aware_recommender.fm
==========================================

.. py:module:: hopwise.model.context_aware_recommender.fm

.. autoapi-nested-parse::

   FM
   ################################################
   Reference:
       Steffen Rendle et al. "Factorization Machines." in ICDM 2010.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.fm.FM


Module Contents
---------------

.. py:class:: FM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   Factorization Machine considers the second-order interaction with features to predict the final score.


   .. py:attribute:: fm


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(interaction)


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



