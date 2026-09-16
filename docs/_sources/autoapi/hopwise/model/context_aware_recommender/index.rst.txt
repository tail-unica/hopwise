hopwise.model.context_aware_recommender
=======================================

.. py:module:: hopwise.model.context_aware_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/context_aware_recommender/afm/index
   /autoapi/hopwise/model/context_aware_recommender/autoint/index
   /autoapi/hopwise/model/context_aware_recommender/dcn/index
   /autoapi/hopwise/model/context_aware_recommender/dcnv2/index
   /autoapi/hopwise/model/context_aware_recommender/deepfm/index
   /autoapi/hopwise/model/context_aware_recommender/dssm/index
   /autoapi/hopwise/model/context_aware_recommender/eulernet/index
   /autoapi/hopwise/model/context_aware_recommender/ffm/index
   /autoapi/hopwise/model/context_aware_recommender/fignn/index
   /autoapi/hopwise/model/context_aware_recommender/fm/index
   /autoapi/hopwise/model/context_aware_recommender/fnn/index
   /autoapi/hopwise/model/context_aware_recommender/fwfm/index
   /autoapi/hopwise/model/context_aware_recommender/kd_dagfm/index
   /autoapi/hopwise/model/context_aware_recommender/lr/index
   /autoapi/hopwise/model/context_aware_recommender/nfm/index
   /autoapi/hopwise/model/context_aware_recommender/pnn/index
   /autoapi/hopwise/model/context_aware_recommender/widedeep/index
   /autoapi/hopwise/model/context_aware_recommender/xdeepfm/index


Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.AFM
   hopwise.model.context_aware_recommender.AutoInt
   hopwise.model.context_aware_recommender.DCN
   hopwise.model.context_aware_recommender.DCNV2
   hopwise.model.context_aware_recommender.DeepFM
   hopwise.model.context_aware_recommender.DSSM
   hopwise.model.context_aware_recommender.EulerNet
   hopwise.model.context_aware_recommender.FFM
   hopwise.model.context_aware_recommender.FiGNN
   hopwise.model.context_aware_recommender.FM
   hopwise.model.context_aware_recommender.FNN
   hopwise.model.context_aware_recommender.FwFM
   hopwise.model.context_aware_recommender.KD_DAGFM
   hopwise.model.context_aware_recommender.LR
   hopwise.model.context_aware_recommender.NFM
   hopwise.model.context_aware_recommender.PNN
   hopwise.model.context_aware_recommender.WideDeep
   hopwise.model.context_aware_recommender.xDeepFM


Package Contents
----------------

.. py:class:: AFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   AFM is a attention based FM model that predict the final score with the attention of input feature.


   .. py:attribute:: attention_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: reg_weight


   .. py:attribute:: num_pair
      :value: 0.0



   .. py:attribute:: attlayer


   .. py:attribute:: p


   .. py:attribute:: dropout_layer


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: build_cross(feat_emb)

      Build the cross feature columns of feature columns

      :param feat_emb: input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
      :type feat_emb: torch.FloatTensor

      :returns:     - torch.FloatTensor: Left part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
                    - torch.FloatTensor: Right part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
      :rtype: tuple



   .. py:method:: afm_layer(infeature)

      Get the attention-based feature interaction score

      :param infeature: input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
      :type infeature: torch.FloatTensor

      :returns: Result of score. shape of [batch_size, 1].
      :rtype: torch.FloatTensor



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



.. py:class:: AutoInt(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   AutoInt is a novel CTR prediction model based on self-attention mechanism,
   which can automatically learn high-order feature interactions in an explicit fashion.



   .. py:attribute:: attention_size


   .. py:attribute:: dropout_probs


   .. py:attribute:: n_layers


   .. py:attribute:: num_heads


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: has_residual


   .. py:attribute:: att_embedding


   .. py:attribute:: embed_output_dim


   .. py:attribute:: atten_output_dim


   .. py:attribute:: mlp_layers


   .. py:attribute:: self_attns


   .. py:attribute:: attn_fc


   .. py:attribute:: deep_predict_layer


   .. py:attribute:: dropout_layer


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: autoint_layer(infeature)

      Get the attention-based feature interaction score

      :param infeature: input feature embedding tensor. shape of[batch_size,field_size,embed_dim].
      :type infeature: torch.FloatTensor

      :returns: Result of score. shape of [batch_size,1] .
      :rtype: torch.FloatTensor



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



.. py:class:: EulerNet(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   EulerNet is a context-based recommendation model.
   It can adaptively learn the arbitrary-order feature interactions in a complex vector space
   by conducting space mapping according to Euler's formula. Meanwhile, it can jointly capture
   the explicit and implicit feature interactions in a unified model architecture.


   .. py:attribute:: Euler_interaction_layers


   .. py:attribute:: mu


   .. py:attribute:: reg


   .. py:attribute:: reg_weight


   .. py:attribute:: sigmoid


   .. py:attribute:: reg_loss


   .. py:attribute:: loss


   .. py:method:: _init_other_weights(module)


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



   .. py:method:: RegularLoss(weight)


.. py:class:: FFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   FFM is a context-based recommendation model. It aims to model the different feature interactions
   between different fields. Each feature has several latent vectors :math:`v_{i,F(j)}`,
   which depend on the field of other features, and one of them is used to do the inner product.

   The model defines as follows:

   .. math::
      y = w_0 + \sum_{i=1}^{m}x_{i}w_{i} + \sum_{i=1}^{m}\sum_{j=i+1}^{m}x_{i}x_{j}<v_{i,F(j)}, v_{j,F(i)}>


   .. py:attribute:: fields


   .. py:attribute:: sigmoid


   .. py:attribute:: feature2id


   .. py:attribute:: feature2field


   .. py:attribute:: feature_names


   .. py:attribute:: feature_dims


   .. py:attribute:: num_fields


   .. py:attribute:: ffm


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: _get_feature2field()

      Create a mapping between features and fields.



   .. py:method:: get_ffm_input(interaction)

      Get different types of ffm layer's input.



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



.. py:class:: FiGNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   FiGNN is a CTR prediction model based on GGNN,
   which can model sophisticated interactions among feature fields on the graph-structured features.


   .. py:attribute:: input_type


   .. py:attribute:: attention_size


   .. py:attribute:: n_layers


   .. py:attribute:: num_heads


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: dropout_layer


   .. py:attribute:: att_embedding


   .. py:attribute:: self_attn


   .. py:attribute:: v_res_embedding


   .. py:attribute:: gnn


   .. py:attribute:: leaky_relu


   .. py:attribute:: W_attn


   .. py:attribute:: gru_cell


   .. py:attribute:: mlp1


   .. py:attribute:: mlp2


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: fignn_layer(in_feature)


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



.. py:class:: FM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   Factorization Machine considers the second-order interaction with features to predict the final score.


   .. py:attribute:: fm


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



.. py:class:: LR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   LR is a context-based recommendation model.
   It aims to predict the CTR given a set of features by using logistic regression,
   which is ideally suited for probabilities as it always predicts a value between 0 and 1:

   .. math::
       CTR = \frac{1}{1+e^{-Z}}

       Z = \sum_{i} {w_i}{x_i}


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



.. py:class:: WideDeep(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   WideDeep is a context-based recommendation model.
   It jointly trains wide linear models and deep neural networks to combine the benefits
   of memorization and generalization for recommender systems. The wide component is a generalized linear model
   of the form :math:`y = w^Tx + b`. The deep component is a feed-forward neural network. The wide component
   and deep component are combined using a weighted sum of their output log odds as the prediction,
   which is then fed to one common logistic loss function for joint training.


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


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



