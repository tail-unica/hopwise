hopwise.model.context_aware_recommender.lr
==========================================

.. py:module:: hopwise.model.context_aware_recommender.lr

.. autoapi-nested-parse::

   LR
   #####################################################
   Reference:
       Matthew Richardson et al. "Predicting Clicks Estimating the Click-Through Rate for New Ads." in WWW 2007.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.lr.LR


Module Contents
---------------

.. py:class:: LR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   LR is a context-based recommendation model.
   It aims to predict the CTR given a set of features by using logistic regression,
   which is ideally suited for probabilities as it always predicts a value between 0 and 1:

   .. math::
       CTR = \frac{1}{1+e^{-Z}}

       Z = \sum_{i} {w_i}{x_i}


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



