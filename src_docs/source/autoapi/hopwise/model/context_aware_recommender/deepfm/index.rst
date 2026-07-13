hopwise.model.context_aware_recommender.deepfm
==============================================

.. py:module:: hopwise.model.context_aware_recommender.deepfm

.. autoapi-nested-parse::

   DeepFM
   ################################################
   Reference:
       Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.deepfm.DeepFM


Module Contents
---------------

.. py:class:: DeepFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
   Also DeepFM can be seen as a combination of FNN and FM.



   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: fm


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



