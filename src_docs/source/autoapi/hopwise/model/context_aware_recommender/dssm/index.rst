hopwise.model.context_aware_recommender.dssm
============================================

.. py:module:: hopwise.model.context_aware_recommender.dssm

.. autoapi-nested-parse::

   DSSM
   ################################################
   Reference:
       PS Huang et al. "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data" in CIKM 2013.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.dssm.DSSM


Module Contents
---------------

.. py:class:: DSSM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   DSSM respectively expresses user and item as low dimensional vectors with mlp layers,
   and uses cosine distance to calculate the distance between the two semantic vectors.



   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: user_feature_num
      :value: 0



   .. py:attribute:: item_feature_num
      :value: 0



   .. py:attribute:: user_mlp_layers


   .. py:attribute:: item_mlp_layers


   .. py:attribute:: loss


   .. py:attribute:: sigmoid


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



