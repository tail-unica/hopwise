hopwise.model.general_recommender.sgl
=====================================

.. py:module:: hopwise.model.general_recommender.sgl

.. autoapi-nested-parse::

   SGL
   ################################################
   Reference:
       Jiancan Wu et al. "SGL: Self-supervised Graph Learning for Recommendation" in SIGIR 2021.

   Reference code:
       https://github.com/wujcan/SGL



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.sgl.SGL


Module Contents
---------------

.. py:class:: SGL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SGL is a GCN-based recommender model.

   SGL supplements the classical supervised task of recommendation with an auxiliary
   self supervised task, which reinforces node representation learning via self-
   discrimination.Specifically,SGL generates multiple views of a node, maximizing the
   agreement between different views of the same node compared to that of other nodes.
   SGL devises three operators to generate the views — node dropout, edge dropout, and
   random walk — that change the graph structure in different manners.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: _user


   .. py:attribute:: _item


   .. py:attribute:: embed_dim


   .. py:attribute:: n_layers


   .. py:attribute:: type


   .. py:attribute:: drop_ratio


   .. py:attribute:: ssl_tau


   .. py:attribute:: reg_weight


   .. py:attribute:: ssl_weight


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: reg_loss


   .. py:attribute:: train_graph


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: graph_construction()

      Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.



   .. py:method:: rand_sample(high, size=None, replace=True)

      Randomly discard some points or edges.

      :param high: Upper limit of index value
      :type high: int
      :param size: Array size after sampling
      :type size: int

      :returns: Array index after sampling, shape: [size]
      :rtype: numpy.ndarray



   .. py:method:: create_adjust_matrix(is_sub: bool)

      Get the normalized interaction matrix of users and items.

      Construct the square matrix from the training data and normalize it
      using the laplace matrix.If it is a subgraph, it may be processed by
      node dropout or edge dropout.

      .. math::
          A_{hat} = D^{-0.5} \times A \times D^{-0.5}

      :returns: csr_matrix of the normalized interaction matrix.



   .. py:method:: csr2tensor(matrix: scipy.sparse.csr_matrix)

      Convert csr_matrix to tensor.

      :param matrix: Sparse matrix to be converted.
      :type matrix: scipy.csr_matrix

      :returns: Transformed sparse matrix.
      :rtype: torch.sparse.FloatTensor



   .. py:method:: forward(graph)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)

      Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

      :param user_emd: Ego embedding of all users after forwarding.
      :type user_emd: torch.Tensor
      :param item_emd: Ego embedding of all items after forwarding.
      :type item_emd: torch.Tensor
      :param user_list: List of the user.
      :type user_list: torch.Tensor
      :param pos_item_list: List of positive examples.
      :type pos_item_list: torch.Tensor
      :param neg_item_list: List of negative examples.
      :type neg_item_list: torch.Tensor

      :returns: Loss of BPR tasks and parameter regularization.
      :rtype: torch.Tensor



   .. py:method:: calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2)

      Calculate the loss of self-supervised tasks.

      :param user_list: List of the user.
      :type user_list: torch.Tensor
      :param pos_item_list: List of positive examples.
      :type pos_item_list: torch.Tensor
      :param user_sub1: Ego embedding of all users in the first subgraph after forwarding.
      :type user_sub1: torch.Tensor
      :param user_sub2: Ego embedding of all users in the second subgraph after forwarding.
      :type user_sub2: torch.Tensor
      :param item_sub1: Ego embedding of all items in the first subgraph after forwarding.
      :type item_sub1: torch.Tensor
      :param item_sub2: Ego embedding of all items in the second subgraph after forwarding.
      :type item_sub2: torch.Tensor

      :returns: Loss of self-supervised tasks.
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



   .. py:method:: train(mode: bool = True)

      Override train method of base class.The subgraph is reconstructed each time it is called.



