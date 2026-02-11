hopwise.model.context_aware_recommender.widedeep
================================================

.. py:module:: hopwise.model.context_aware_recommender.widedeep

.. autoapi-nested-parse::

   WideDeep
   #####################################################
   Reference:
       Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems." in RecSys 2016.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.widedeep.WideDeep


Module Contents
---------------

.. py:class:: WideDeep(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   WideDeep is a context-based recommendation model.
   It jointly trains wide linear models and deep neural networks to combine the benefits
   of memorization and generalization for recommender systems. The wide component is a generalized linear model
   of the form :math:`y = w^Tx + b`. The deep component is a feed-forward neural network. The wide component
   and deep component are combined using a weighted sum of their output log odds as the prediction,
   which is then fed to one common logistic loss function for joint training.


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: mlp_layers


   .. py:attribute:: deep_predict_layer


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



