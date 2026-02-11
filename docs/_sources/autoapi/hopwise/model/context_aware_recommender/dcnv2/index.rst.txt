hopwise.model.context_aware_recommender.dcnv2
=============================================

.. py:module:: hopwise.model.context_aware_recommender.dcnv2

.. autoapi-nested-parse::

   DCN V2
   ################################################
   Reference:
       Ruoxi Wang at al. "Dcn v2: Improved deep & cross network and practical lessons for web-scale
       learning to rank systems." in WWW 2021.

   Reference code:
       https://github.com/shenweichen/DeepCTR-Torch



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.dcnv2.DCNV2


Module Contents
---------------

.. py:class:: DCNV2(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   DCNV2 improves the cross network by extending the original weight vector to a matrix,
   significantly improves the expressiveness of DCN. It also introduces the MoE and
   low rank techniques to reduce time cost.


   .. py:attribute:: mixed


   .. py:attribute:: structure


   .. py:attribute:: cross_layer_num


   .. py:attribute:: embedding_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: reg_weight


   .. py:attribute:: dropout_prob


   .. py:attribute:: in_feature_num


   .. py:attribute:: bias


   .. py:attribute:: mlp_layers


   .. py:attribute:: reg_loss


   .. py:attribute:: sigmoid


   .. py:attribute:: tanh


   .. py:attribute:: softmax


   .. py:attribute:: loss


   .. py:method:: cross_network(x_0)

      Cross network is composed of cross layers, with each layer having the following formula.

      .. math:: x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l

      :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
      (l + 1)-th cross layers, respectively.
      :math:`W_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

      :param x_0: Embedding vectors of all features, input of cross network.
      :type x_0: torch.Tensor

      :returns: output of cross network, [batch_size, num_feature_field * embedding_size]
      :rtype: torch.Tensor



   .. py:method:: cross_network_mix(x_0)

      Cross network part of DCN-mix, which add MoE and nonlinear transformation in low-rank space.

      .. math::
          x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
      .. math::
          E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)

      :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
      :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
      :math:`g` is the nonlinear activation function.

      :param x_0: Embedding vectors of all features, input of cross network.
      :type x_0: torch.Tensor

      :returns: output of mixed cross network, [batch_size, num_feature_field * embedding_size]
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



