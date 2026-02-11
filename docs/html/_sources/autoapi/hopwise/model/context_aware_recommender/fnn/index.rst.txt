hopwise.model.context_aware_recommender.fnn
===========================================

.. py:module:: hopwise.model.context_aware_recommender.fnn

.. autoapi-nested-parse::

   FNN
   ################################################
   Reference:
       Weinan Zhang1 et al. "Deep Learning over Multi-field Categorical Data" in ECIR 2016



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.fnn.FNN


Module Contents
---------------

.. py:class:: FNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   FNN which also called DNN is a basic version of CTR model that use mlp from field features to predict score.

   .. note::

      Based on the experiments in the paper above, This implementation incorporate
      Dropout instead of L2 normalization to relieve over-fitting.
      Our implementation of FNN is a basic version without pretrain support.
      If you want to pretrain the feature embedding as the original paper,
      we suggest you to construct a advanced FNN model and train it in two-stage
      process with our FM model.


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: mlp_layers


   .. py:attribute:: predict_layer


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



