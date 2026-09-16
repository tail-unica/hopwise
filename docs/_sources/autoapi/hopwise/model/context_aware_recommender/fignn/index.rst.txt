hopwise.model.context_aware_recommender.fignn
=============================================

.. py:module:: hopwise.model.context_aware_recommender.fignn

.. autoapi-nested-parse::

   FiGNN
   ################################################
   Reference:
       Li, Zekun, et al.  "Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction"
       in CIKM 2019.

   Reference code:
       - https://github.com/CRIPAC-DIG/GraphCTR
       - https://github.com/xue-pai/FuxiCTR



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.fignn.GraphLayer
   hopwise.model.context_aware_recommender.fignn.FiGNN


Module Contents
---------------

.. py:class:: GraphLayer(num_fields, embedding_size)

   Bases: :py:obj:`torch.nn.Module`


   The implementations of the GraphLayer part and the Attentional Edge Weights part are adapted from https://github.com/xue-pai/FuxiCTR.


   .. py:attribute:: W_in


   .. py:attribute:: W_out


   .. py:attribute:: bias_p


   .. py:method:: forward(g, h)


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



