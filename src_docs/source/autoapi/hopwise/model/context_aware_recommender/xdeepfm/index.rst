hopwise.model.context_aware_recommender.xdeepfm
===============================================

.. py:module:: hopwise.model.context_aware_recommender.xdeepfm

.. autoapi-nested-parse::

   xDeepFM
   ################################################
   Reference:
       Jianxun Lian at al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems."
       in SIGKDD 2018.

   Reference code:
       - https://github.com/Leavingseason/xDeepFM
       - https://github.com/shenweichen/DeepCTR-Torch



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.xdeepfm.xDeepFM


Module Contents
---------------

.. py:class:: xDeepFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   xDeepFM combines a CIN (Compressed Interaction Network) with a classical DNN.
   The model is able to learn certain bounded-degree feature interactions explicitly;
   Besides, it can also learn arbitrary low- and high-order feature interactions implicitly.


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: reg_weight


   .. py:attribute:: dropout_prob


   .. py:attribute:: direct


   .. py:attribute:: conv1d_list


   .. py:attribute:: field_nums


   .. py:attribute:: mlp_layers


   .. py:attribute:: cin_linear


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: reg_loss(parameters)

      Calculate the L2 normalization loss of parameters in a certain layer.

      :returns: The L2 Loss tensor. shape of [1,]
      :rtype: loss(torch.FloatTensor)



   .. py:method:: calculate_reg_loss()

      Calculate the final L2 normalization loss of model parameters.
      Including weight matrices of mlp layers, linear layer and convolutional layers.

      :returns: The L2 Loss tensor. shape of [1,]
      :rtype: loss(torch.FloatTensor)



   .. py:method:: compressed_interaction_network(input_features, activation='ReLU')

      For k-th CIN layer, the output :math:`X_k` is calculated via

      .. math::
          x_{h,*}^{k} = \sum_{i=1}^{H_k-1} \sum_{j=1}^{m}W_{i,j}^{k,h}(X_{i,*}^{k-1} \circ x_{j,*}^0)

      :math:`H_k` donates the number of feature vectors in the k-th layer,
      :math:`1 \le h \le H_k`.
      :math:`\circ` donates the Hadamard product.

      And Then, We apply sum pooling on each feature map of the hidden layer.
      Finally, All pooling vectors from hidden layers are concatenated.

      :param input_features: [batch_size, field_num, embed_dim]. Embedding vectors of all features.
      :type input_features: torch.Tensor
      :param activation: name of activation function.
      :type activation: str

      :returns: [batch_size, num_feature_field * embedding_size]. output of CIN layer.
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



