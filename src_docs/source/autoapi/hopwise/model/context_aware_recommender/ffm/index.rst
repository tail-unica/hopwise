hopwise.model.context_aware_recommender.ffm
===========================================

.. py:module:: hopwise.model.context_aware_recommender.ffm

.. autoapi-nested-parse::

   FFM
   #####################################################
   Reference:
       Yuchin Juan et al. "Field-aware Factorization Machines for CTR Prediction" in RecSys 2016.

   Reference code:
       https://github.com/rixwew/pytorch-fm



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.ffm.FFM
   hopwise.model.context_aware_recommender.ffm.FieldAwareFactorizationMachine


Module Contents
---------------

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



.. py:class:: FieldAwareFactorizationMachine(feature_names, feature_dims, feature2id, feature2field, num_fields, embed_dim, device)

   Bases: :py:obj:`torch.nn.Module`


   This is Field-Aware Factorization Machine Module for FFM.


   .. py:attribute:: token_feature_names


   .. py:attribute:: float_feature_names


   .. py:attribute:: token_seq_feature_names


   .. py:attribute:: float_seq_feature_names


   .. py:attribute:: token_feature_dims


   .. py:attribute:: float_feature_dims


   .. py:attribute:: token_seq_feature_dims


   .. py:attribute:: float_seq_feature_dims


   .. py:attribute:: feature2id


   .. py:attribute:: feature2field


   .. py:attribute:: num_features


   .. py:attribute:: num_fields


   .. py:attribute:: embed_dim


   .. py:attribute:: device


   .. py:method:: forward(input_x)

      Model the different interaction strengths of different field pairs.

      :param input_x: (token_ffm_input, float_ffm_input, token_seq_ffm_input)

                      token_ffm_input (torch.cuda.FloatTensor): [batch_size, num_token_features] or None

                      float_ffm_input (torch.cuda.FloatTensor): [batch_size, num_float_features] or None

                      token_seq_ffm_input (list): length is num_token_seq_features or 0
      :type input_x: a tuple

      :returns: The results of all features' field-aware interactions.
                shape: [batch_size, num_fields, emb_dim]
      :rtype: torch.cuda.FloatTensor



   .. py:method:: _get_input_x_emb(token_input_x_emb, float_input_x_emb, token_seq_input_x_emb, float_seq_input_x_emb)


   .. py:method:: _emb_token_ffm_input(token_ffm_input)


   .. py:method:: _emb_float_ffm_input(float_ffm_input)


   .. py:method:: _emb_token_seq_ffm_input(token_seq_ffm_input)


   .. py:method:: _emb_float_seq_ffm_input(float_seq_ffm_input)


