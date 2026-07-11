hopwise.model.context_aware_recommender.kd_dagfm
================================================

.. py:module:: hopwise.model.context_aware_recommender.kd_dagfm

.. autoapi-nested-parse::

   KD_DAGFM
   ################################################
   Reference:
       Zhen Tian et al. "Directed Acyclic Graph Factorization Machines for CTR Prediction via Knowledge Distillation."
       in WSDM 2023.
   Reference code:
       https://github.com/chenyuwuxin/DAGFM



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.kd_dagfm.KD_DAGFM
   hopwise.model.context_aware_recommender.kd_dagfm.DAGFM
   hopwise.model.context_aware_recommender.kd_dagfm.CrossNet
   hopwise.model.context_aware_recommender.kd_dagfm.CINComp
   hopwise.model.context_aware_recommender.kd_dagfm.CIN


Module Contents
---------------

.. py:class:: KD_DAGFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   KD_DAGFM is a context-based recommendation model. The model is based on directed acyclic graph and knowledge
   distillation. It can learn arbitrary feature interactions from the complex teacher networks and achieve
   approximately lossless model performance. It can also greatly reduce the computational resource costs.


   .. py:attribute:: phase


   .. py:attribute:: alpha


   .. py:attribute:: beta


   .. py:attribute:: student_network


   .. py:attribute:: teacher_network


   .. py:attribute:: loss_fn


   .. py:method:: get_teacher_config(config)


   .. py:method:: FeatureInteraction(feature)


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



.. py:class:: DAGFM(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: type


   .. py:attribute:: depth


   .. py:attribute:: adj_matrix


   .. py:attribute:: connect_layer


   .. py:attribute:: linear


   .. py:method:: FeatureInteraction(feature)


.. py:class:: CrossNet(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: depth


   .. py:attribute:: embedding_size


   .. py:attribute:: feature_num


   .. py:attribute:: in_feature_num


   .. py:attribute:: cross_layer_w


   .. py:attribute:: bias


   .. py:attribute:: linear


   .. py:method:: FeatureInteraction(x_0)


   .. py:method:: forward(feature)


.. py:class:: CINComp(indim, outdim, config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: conv


   .. py:method:: forward(feature, base)


.. py:class:: CIN(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: cinlist


   .. py:attribute:: cin


   .. py:attribute:: linear


   .. py:attribute:: backbone
      :value: ['cin', 'linear']



   .. py:attribute:: loss_fn


   .. py:method:: FeatureInteraction(feature)


   .. py:method:: forward(feature)


