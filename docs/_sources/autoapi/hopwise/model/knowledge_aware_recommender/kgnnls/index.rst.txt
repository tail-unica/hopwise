hopwise.model.knowledge_aware_recommender.kgnnls
================================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kgnnls

.. autoapi-nested-parse::

   KGNNLS
   ################################################

   Reference:
       Hongwei Wang et al. "Knowledge-aware Graph Neural Networks with Label Smoothness Regularization
       for Recommender Systems." in KDD 2019.

   Reference code:
       https://github.com/hwwang55/KGNN-LS



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kgnnls.KGNNLS


Module Contents
---------------

.. py:class:: KGNNLS(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGNN-LS is a knowledge-based recommendation model.
   KGNN-LS transforms the knowledge graph into a user-specific weighted graph and then apply a graph neural network to
   compute personalized item embeddings. To provide better inductive bias, KGNN-LS relies on label smoothness
   assumption, which posits that adjacent items in the knowledge graph are likely to have similar user relevance
   labels/scores. Label smoothness provides regularization over the edge weights and it is equivalent  to a label
   propagation scheme on a graph.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: neighbor_sample_size


   .. py:attribute:: aggregator_class


   .. py:attribute:: n_iter


   .. py:attribute:: reg_weight


   .. py:attribute:: ls_weight


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: interaction_table


   .. py:attribute:: softmax


   .. py:attribute:: linear_layers


   .. py:attribute:: ReLU


   .. py:attribute:: Tanh


   .. py:attribute:: bce_loss


   .. py:attribute:: l2_loss


   .. py:attribute:: other_parameter_name
      :value: ['adj_entity', 'adj_relation']



   .. py:method:: get_interaction_table(user_id, item_id, y)

      Get interaction_table that is used for fetching user-item interaction label in LS regularization.

      :param user_id: the user id in user-item interactions, shape: [n_interactions, 1]
      :type user_id: torch.Tensor
      :param item_id: the item id in user-item interactions, shape: [n_interactions, 1]
      :type item_id: torch.Tensor
      :param y: the label in user-item interactions, shape: [n_interactions, 1]
      :type y: torch.Tensor

      :returns:     - interaction_table(dict): key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
                    - offset(int): The offset that is used for calculating the key(index) in interaction_table
      :rtype: tuple



   .. py:method:: sample_neg_interaction(pos_interaction_table, offset)

      Sample neg_interaction to construct train data.

      :param pos_interaction_table: the interaction_table that only contains pos_interaction.
      :type pos_interaction_table: dict
      :param offset: The offset that is used for calculating the key(index) in interaction_table
      :type offset: int

      :returns: key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
      :rtype: interaction_table(dict)



   .. py:method:: construct_adj(kg_graph)

      Get neighbors and corresponding relations for each entity in the KG.

      :param kg_graph: an undirected graph
      :type kg_graph: scipy.sparse.coo_matrix

      :returns:

                    - adj_entity (torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                      shape: [n_entities, neighbor_sample_size]
                    - adj_relation (torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                      shape: [n_entities, neighbor_sample_size]
      :rtype: tuple



   .. py:method:: get_neighbors(items)

      Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

      :param items: The input tensor that contains item's id, shape: [batch_size, ]
      :type items: torch.LongTensor

      :returns:

                    - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                      dimensions of entities: {[batch_size, 1],
                      [batch_size, n_neighbor],
                      [batch_size, n_neighbor^2],
                      ...,
                      [batch_size, n_neighbor^n_iter]}
                    - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                      entities. Relations have the same shape as entities.
      :rtype: tuple



   .. py:method:: aggregate(user_embeddings, entities, relations)

      For each item, aggregate the entity representation and its neighborhood representation into a single vector.

      :param user_embeddings: The embeddings of users, shape: [batch_size, embedding_size]
      :type user_embeddings: torch.FloatTensor
      :param entities: entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                       dimensions of entities: {[batch_size, 1],
                       [batch_size, n_neighbor],
                       [batch_size, n_neighbor^2],
                       ...,
                       [batch_size, n_neighbor^n_iter]}
      :type entities: list
      :param relations: relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                        relations have the same shape as entities.
      :type relations: list

      :returns: The embeddings of items, shape: [batch_size, embedding_size]
      :rtype: item_embeddings(torch.FloatTensor)



   .. py:method:: label_smoothness_predict(user_embeddings, user, entities, relations)

      Predict the label of items by label smoothness.

      :param user_embeddings: The embeddings of users, shape: [batch_size*2, embedding_size],
      :type user_embeddings: torch.FloatTensor
      :param user: the index of users, shape: [batch_size*2]
      :type user: torch.FloatTensor
      :param entities: entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                       dimensions of entities: {[batch_size*2, 1],
                       [batch_size*2, n_neighbor],
                       [batch_size*2, n_neighbor^2],
                       ...,
                       [batch_size*2, n_neighbor^n_iter]}
      :type entities: list
      :param relations: relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                        relations have the same shape as entities.
      :type relations: list

      :returns: The predicted label of items, shape: [batch_size*2]
      :rtype: predicted_labels(torch.FloatTensor)



   .. py:method:: forward(user, item)


   .. py:method:: calculate_ls_loss(user, item, target)

      Calculate label smoothness loss.

      :param user: the index of users, shape: [batch_size*2],
      :type user: torch.FloatTensor
      :param item: the index of items, shape: [batch_size*2],
      :type item: torch.FloatTensor
      :param target: the label of user-item, shape: [batch_size*2],
      :type target: torch.FloatTensor

      :returns: label smoothness loss
      :rtype: ls_loss



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



