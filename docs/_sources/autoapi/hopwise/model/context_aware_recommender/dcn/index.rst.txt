hopwise.model.context_aware_recommender.dcn
===========================================

.. py:module:: hopwise.model.context_aware_recommender.dcn

.. autoapi-nested-parse::

   DCN
   ################################################
   Reference:
       Ruoxi Wang at al. "Deep & Cross Network for Ad Click Predictions." in ADKDD 2017.

   Reference code:
       https://github.com/shenweichen/DeepCTR-Torch



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.dcn.DCN


Module Contents
---------------

.. py:class:: DCN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
   automatically construct limited high-degree cross features, and learns the corresponding weights.



   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: cross_layer_num


   .. py:attribute:: reg_weight


   .. py:attribute:: dropout_prob


   .. py:attribute:: cross_layer_w


   .. py:attribute:: cross_layer_b


   .. py:attribute:: mlp_layers


   .. py:attribute:: predict_layer


   .. py:attribute:: reg_loss


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: cross_network(x_0)

      Cross network is composed of cross layers, with each layer having the following formula.

      .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

      :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
      (l + 1)-th cross layers, respectively.
      :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

      :param x_0: Embedding vectors of all features, input of cross network.
      :type x_0: torch.Tensor

      :returns: output of cross network, [batch_size, num_feature_field * embedding_size]
      :rtype: torch.Tensor



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



