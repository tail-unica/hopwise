hopwise.model.general_recommender.gcmc
======================================

.. py:module:: hopwise.model.general_recommender.gcmc

.. autoapi-nested-parse::

   GCMC
   ################################################

   Reference:
       van den Berg et al. "Graph Convolutional Matrix Completion." in SIGKDD 2018.

   Reference code:
       https://github.com/riannevdberg/gc-mc



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.gcmc.GCMC
   hopwise.model.general_recommender.gcmc.GcEncoder
   hopwise.model.general_recommender.gcmc.BiDecoder


Functions
---------

.. autoapisummary::

   hopwise.model.general_recommender.gcmc.orthogonal


Module Contents
---------------

.. py:class:: GCMC(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   GCMC is a model that incorporate graph autoencoders for recommendation.

   Graph autoencoders are comprised of:

   1) a graph encoder model :math:`Z = f(X; A)`, which take as input an :math:`N \times D` feature matrix X and
   a graph adjacency matrix A, and produce an :math:`N \times E` node embedding matrix
   :math:`Z = [z_1^T,..., z_N^T ]^T`;

   2) a pairwise decoder model :math:`\hat A = g(Z)`, which takes pairs of node embeddings :math:`(z_i, z_j)` and
   predicts respective entries :math:`\hat A_{ij}` in the adjacency matrix.

   Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
   and :math:`E` the embedding size.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: num_all


   .. py:attribute:: dropout_prob


   .. py:attribute:: sparse_feature


   .. py:attribute:: gcn_output_dim


   .. py:attribute:: dense_output_dim


   .. py:attribute:: n_class


   .. py:attribute:: num_basis_functions


   .. py:attribute:: input_dim


   .. py:attribute:: Graph


   .. py:attribute:: support


   .. py:attribute:: accum


   .. py:attribute:: GcEncoder


   .. py:attribute:: BiDecoder


   .. py:attribute:: loss_function


   .. py:method:: forward(user_X, item_X, user, item)


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



.. py:class:: GcEncoder(accum, num_user, num_item, support, input_dim, gcn_output_dim, dense_output_dim, drop_prob, device, sparse_feature=True, act_dense=lambda x: x, share_user_item_weights=True, bias=False)

   Bases: :py:obj:`torch.nn.Module`


   Graph Convolutional Encoder
   GcEncoder take as input an :math:`N \times D` feature matrix :math:`X` and a graph adjacency matrix :math:`A`,
   and produce an :math:`N \times E` node embedding matrix;
   Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
   and :math:`E` the embedding size.


   .. py:attribute:: num_users


   .. py:attribute:: num_items


   .. py:attribute:: input_dim


   .. py:attribute:: gcn_output_dim


   .. py:attribute:: dense_output_dim


   .. py:attribute:: accum


   .. py:attribute:: sparse_feature
      :value: True



   .. py:attribute:: device


   .. py:attribute:: dropout_prob


   .. py:attribute:: dropout


   .. py:attribute:: dense_activate


   .. py:attribute:: activate


   .. py:attribute:: share_weights
      :value: True



   .. py:attribute:: bias
      :value: False



   .. py:attribute:: support


   .. py:attribute:: num_support


   .. py:attribute:: dense_layer_u


   .. py:method:: _init_weights()


   .. py:method:: forward(user_X, item_X)


.. py:class:: BiDecoder(input_dim, output_dim, drop_prob, device, num_weights=3, act=lambda x: x)

   Bases: :py:obj:`torch.nn.Module`


   Bi-linear decoder
   BiDecoder takes pairs of node embeddings and predicts respective entries in the adjacency matrix.


   .. py:attribute:: input_dim


   .. py:attribute:: output_dim


   .. py:attribute:: num_weights
      :value: 3



   .. py:attribute:: device


   .. py:attribute:: activate


   .. py:attribute:: dropout_prob


   .. py:attribute:: dropout


   .. py:attribute:: weights


   .. py:attribute:: dense_layer


   .. py:method:: _init_weights()


   .. py:method:: forward(u_inputs, i_inputs, users, items=None)


.. py:function:: orthogonal(shape, scale=1.1)

   Initialization function for weights in class GCMC.
   From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120


