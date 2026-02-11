hopwise.model.context_aware_recommender.autoint
===============================================

.. py:module:: hopwise.model.context_aware_recommender.autoint

.. autoapi-nested-parse::

   AutoInt
   ################################################
   Reference:
       Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
       in CIKM 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.autoint.AutoInt


Module Contents
---------------

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



