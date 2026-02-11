hopwise.model.sequential_recommender.gcsan
==========================================

.. py:module:: hopwise.model.sequential_recommender.gcsan

.. autoapi-nested-parse::

   GCSAN
   ################################################

   Reference:
       Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.gcsan.GNN
   hopwise.model.sequential_recommender.gcsan.GCSAN


Module Contents
---------------

.. py:class:: GNN(embedding_size, step=1)

   Bases: :py:obj:`torch.nn.Module`


   Graph neural networks are well-suited for session-based recommendation,
   because it can automatically extract features of session graphs with considerations of rich node connections.


   .. py:attribute:: step
      :value: 1



   .. py:attribute:: embedding_size


   .. py:attribute:: input_size


   .. py:attribute:: gate_size


   .. py:attribute:: w_ih


   .. py:attribute:: w_hh


   .. py:attribute:: b_ih


   .. py:attribute:: b_hh


   .. py:attribute:: linear_edge_in


   .. py:attribute:: linear_edge_out


   .. py:method:: _reset_parameters()


   .. py:method:: GNNCell(A, hidden)

      Obtain latent vectors of nodes via gated graph neural network.

      :param A: The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]
      :type A: torch.FloatTensor
      :param hidden: The item node embedding matrix, shape of
                     [batch_size, max_session_len, embedding_size]
      :type hidden: torch.FloatTensor

      :returns: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: forward(A, hidden)


.. py:class:: GCSAN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   GCSAN captures rich local dependencies via graph neural network,
    and learns long-range dependencies by applying the self-attention mechanism.

   .. note::

      In the original paper, the attention mechanism in the self-attention layer is a single head,
      for the reusability of the project code, we use a unified transformer component.
      According to the experimental results, we only applied regularization to embedding.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: step


   .. py:attribute:: device


   .. py:attribute:: weight


   .. py:attribute:: reg_weight


   .. py:attribute:: loss_type


   .. py:attribute:: initializer_range


   .. py:attribute:: item_embedding


   .. py:attribute:: gnn


   .. py:attribute:: self_attention


   .. py:attribute:: reg_loss


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _get_slice(item_seq)


   .. py:method:: forward(item_seq, item_seq_len)


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



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



