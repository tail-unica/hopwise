hopwise.model.context_aware_recommender.fwfm
============================================

.. py:module:: hopwise.model.context_aware_recommender.fwfm

.. autoapi-nested-parse::

   FwFM
   #####################################################
   Reference:
       Junwei Pan et al. "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising."
       in WWW 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.fwfm.FwFM


Module Contents
---------------

.. py:class:: FwFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   FwFM is a context-based recommendation model. It aims to model the different feature interactions
   between different fields in a much more memory-efficient way. It proposes a field pair weight matrix
   :math:`r_{F(i),F(j)}`, to capture the heterogeneity of field pair interactions.

   The model defines as follows:

   .. math::
      y = w_0 + \sum_{i=1}^{m}x_{i}w_{i} + \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}


   .. py:attribute:: dropout_prob


   .. py:attribute:: fields


   .. py:attribute:: num_features
      :value: 0



   .. py:attribute:: dropout_layer


   .. py:attribute:: sigmoid


   .. py:attribute:: feature2id


   .. py:attribute:: feature2field


   .. py:attribute:: feature_names


   .. py:attribute:: feature_dims


   .. py:attribute:: num_fields


   .. py:attribute:: num_pair


   .. py:attribute:: weight


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: _get_feature2field()

      Create a mapping between features and fields.



   .. py:method:: fwfm_layer(infeature)

      Get the field pair weight matrix r_{F(i),F(j)}, and model the different interaction strengths of
      different field pairs :math:`\sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i}, v_{j}>r_{F(i),F(j)}`.

      :param infeature: [batch_size, field_size, embed_dim]
      :type infeature: torch.cuda.FloatTensor

      :returns: [batch_size, 1]
      :rtype: torch.cuda.FloatTensor



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



