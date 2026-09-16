hopwise.model.context_aware_recommender.pnn
===========================================

.. py:module:: hopwise.model.context_aware_recommender.pnn

.. autoapi-nested-parse::

   PNN
   ################################################
   Reference:
       Qu Y et al. "Product-based neural networks for user response prediction." in ICDM 2016

   Reference code:
       - https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/pnn.py
       - https://github.com/Atomu2014/product-nets/blob/master/python/models.py



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.pnn.PNN
   hopwise.model.context_aware_recommender.pnn.InnerProductLayer
   hopwise.model.context_aware_recommender.pnn.OuterProductLayer


Module Contents
---------------

.. py:class:: PNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   PNN calculate inner and outer product of feature embedding.
   You can choose the product option with the parameter of use_inner and use_outer



   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: use_inner


   .. py:attribute:: use_outer


   .. py:attribute:: reg_weight


   .. py:attribute:: num_pair
      :value: 0



   .. py:attribute:: mlp_layers


   .. py:attribute:: predict_layer


   .. py:attribute:: relu


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: reg_loss()

      Calculate the L2 normalization loss of model parameters.
      Including weight matrices of mlp layers.

      :returns: The L2 Loss tensor. shape of [1,]
      :rtype: loss(torch.FloatTensor)



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



.. py:class:: InnerProductLayer(num_feature_field, device)

   Bases: :py:obj:`torch.nn.Module`


   InnerProduct Layer used in PNN that compute the element-wise
   product or inner product between feature vectors.



   .. py:attribute:: num_feature_field


   .. py:method:: forward(feat_emb)

      :param feat_emb: 3D tensor with shape: [batch_size,num_pairs,embedding_size].
      :type feat_emb: torch.FloatTensor

      :returns: The inner product of input tensor. shape of [batch_size, num_pairs]
      :rtype: inner_product(torch.FloatTensor)



.. py:class:: OuterProductLayer(num_feature_field, embedding_size, device)

   Bases: :py:obj:`torch.nn.Module`


   OuterProduct Layer used in PNN. This implementation is
   adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.


   .. py:attribute:: num_feature_field


   .. py:attribute:: kernel


   .. py:method:: forward(feat_emb)

      :param feat_emb: 3D tensor with shape: [batch_size,num_pairs,embedding_size].
      :type feat_emb: torch.FloatTensor

      :returns: The outer product of input tensor. shape of [batch_size, num_pairs]
      :rtype: outer_product(torch.FloatTensor)



