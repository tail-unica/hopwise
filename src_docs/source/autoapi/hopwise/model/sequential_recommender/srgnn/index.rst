hopwise.model.sequential_recommender.srgnn
==========================================

.. py:module:: hopwise.model.sequential_recommender.srgnn

.. autoapi-nested-parse::

   SRGNN
   ################################################

   Reference:
       Shu Wu et al. "Session-based Recommendation with Graph Neural Networks." in AAAI 2019.

   Reference code:
       https://github.com/CRIPAC-DIG/SR-GNN



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.srgnn.GNN
   hopwise.model.sequential_recommender.srgnn.SRGNN


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


   .. py:attribute:: b_iah


   .. py:attribute:: b_ioh


   .. py:attribute:: linear_edge_in


   .. py:attribute:: linear_edge_out


   .. py:method:: GNNCell(A, hidden)

      Obtain latent vectors of nodes via graph neural networks.

      :param A: The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]
      :type A: torch.FloatTensor
      :param hidden: The item node embedding matrix, shape of
                     [batch_size, max_session_len, embedding_size]
      :type hidden: torch.FloatTensor

      :returns: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: forward(A, hidden)


.. py:class:: SRGNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SRGNN regards the conversation history as a directed graph.
   In addition to considering the connection between the item and the adjacent item,
   it also considers the connection with other interactive items.

   Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

   Outgoing edges:
       === ===== ===== ===== =====
        \    1     2     3     4
       === ===== ===== ===== =====
        1    0     1     0     0
        2    0     0    1/2   1/2
        3    0     1     0     0
        4    0     0     0     0
       === ===== ===== ===== =====

   Incoming edges:
       === ===== ===== ===== =====
        \    1     2     3     4
       === ===== ===== ===== =====
        1    0     0     0     0
        2   1/2    0    1/2    0
        3    0     1     0     0
        4    0     1     0     0
       === ===== ===== ===== =====


   .. py:attribute:: embedding_size


   .. py:attribute:: step


   .. py:attribute:: device


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: gnn


   .. py:attribute:: linear_one


   .. py:attribute:: linear_two


   .. py:attribute:: linear_three


   .. py:attribute:: linear_transform


   .. py:method:: _reset_parameters()


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



