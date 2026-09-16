hopwise.model.general_recommender.nncf
======================================

.. py:module:: hopwise.model.general_recommender.nncf

.. autoapi-nested-parse::

   NNCF
   ################################################
   Reference:
       Ting Bai et al. "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." in CIKM 2017.

   Reference code:
       https://github.com/Tbbaby/NNCF-Pytorch



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.nncf.NNCF


Module Contents
---------------

.. py:class:: NNCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NNCF is an neural network enhanced matrix factorization model which also captures neighborhood information.
   We implement the NNCF model with three ways to process neighborhood information.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: interaction_matrix


   .. py:attribute:: ui_embedding_size


   .. py:attribute:: neigh_embedding_size


   .. py:attribute:: num_conv_kernel


   .. py:attribute:: conv_kernel_size


   .. py:attribute:: pool_kernel_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: neigh_num


   .. py:attribute:: neigh_info_method


   .. py:attribute:: resolution


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: user_neigh_embedding


   .. py:attribute:: item_neigh_embedding


   .. py:attribute:: user_conv


   .. py:attribute:: item_conv


   .. py:attribute:: mlp_layers


   .. py:attribute:: out_layer


   .. py:attribute:: dropout_layer


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: Max_ner(lst, max_ner)

      Unify embedding length of neighborhood information for efficiency consideration.
      Truncate the list if the length is larger than max_ner.
      Otherwise, pad it with 0.

      :param lst: The input list contains node's neighbors.
      :type lst: list
      :param max_ner: The number of neighbors we choose for each node.
      :type max_ner: int

      :returns: The list of a node's community neighbors.
      :rtype: list



   .. py:method:: get_community_member(partition, community_dict, node, kind)

      Find other nodes in the same community.
      e.g. If the node starts with letter "i",
      the other nodes start with letter "i" in the same community dict group are its community neighbors.

      :param partition: The input dict that contains the community each node belongs.
      :type partition: dict
      :param community_dict: The input dict that shows the nodes each community contains.
      :type community_dict: dict
      :param node: The id of the input node.
      :type node: int
      :param kind: The type of the input node.
      :type kind: char

      :returns: The list of a node's community neighbors.
      :rtype: list



   .. py:method:: prepare_vector_element(partition, relation, community_dict)

      Find the community neighbors of each node, i.e. I(u) and U(i).
      Then reset the id of nodes.

      :param partition: The input dict that contains the community each node belongs.
      :type partition: dict
      :param relation: The input list that contains the relationships of users and items.
      :type relation: list
      :param community_dict: The input dict that shows the nodes each community contains.
      :type community_dict: dict

      :returns: The list of nodes' community neighbors.
      :rtype: list



   .. py:method:: get_neigh_louvain()

      Get neighborhood information using louvain algorithm.
      First, change the id of node,
      for example, the id of user node "1" will be set to "u_1" in order to use louvain algorithm.
      Second, use louvain algorithm to seperate nodes into different communities.
      Finally, find the community neighbors of each node with the same type and reset the id of the nodes.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_knn()

      Get neighborhood information using knn algorithm.
      Find direct neighbors of each node, if the number of direct neighbors is less than neigh_num,
      add other similar neighbors using knn algorithm.
      Otherwise, select random top k direct neighbors, k equals to the number of neighbors.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_random()

      Get neighborhood information using random algorithm.
      Select random top k direct neighbors, k equals to the number of neighbors.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_info(user, item)

      Get a batch of neighborhood embedding tensor according to input id.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor
      :param item: The input tensor that contains item's id, shape: [batch_size, ]
      :type item: torch.LongTensor

      :returns: The neighborhood embedding tensor of a batch of user, shape: [batch_size, neigh_embedding_size]
                torch.FloatTensor: The neighborhood embedding tensor of a batch of item, shape: [batch_size, neigh_embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: forward(user, item)


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



