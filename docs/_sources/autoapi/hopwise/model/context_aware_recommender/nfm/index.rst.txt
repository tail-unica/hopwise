hopwise.model.context_aware_recommender.nfm
===========================================

.. py:module:: hopwise.model.context_aware_recommender.nfm

.. autoapi-nested-parse::

   NFM
   ################################################
   Reference:
       He X, Chua T S. "Neural factorization machines for sparse predictive analytics" in SIGIR 2017



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.nfm.NFM


Module Contents
---------------

.. py:class:: NFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   NFM replace the fm part as a mlp to model the feature interaction.


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: fm


   .. py:attribute:: bn


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



