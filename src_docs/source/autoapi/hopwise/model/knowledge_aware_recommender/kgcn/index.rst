hopwise.model.knowledge_aware_recommender.kgcn
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kgcn

.. autoapi-nested-parse::

   KGCN
   ################################################

   Reference:
       Hongwei Wang et al. "Knowledge graph convolution networks for recommender systems." in WWW 2019.

   Reference code:
       https://github.com/hwwang55/KGCN



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kgcn.KGCN


Module Contents
---------------

.. py:class:: KGCN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGCN is a knowledge-based recommendation model that captures inter-item relatedness effectively by mining their
   associated attributes on the KG. To automatically discover both high-order structure information and semantic
   information of the KG, we treat KG as an undirected graph and sample from the neighbors for each entity in the KG
   as their receptive field, then combine neighborhood information with bias when calculating the representation of a
   given entity.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_iter


   .. py:attribute:: aggregator_class


   .. py:attribute:: reg_weight


   .. py:attribute:: neighbor_sample_size


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: softmax


   .. py:attribute:: linear_layers


   .. py:attribute:: ReLU


   .. py:attribute:: Tanh


   .. py:attribute:: bce_loss


   .. py:attribute:: l2_loss


   .. py:attribute:: other_parameter_name
      :value: ['adj_entity', 'adj_relation']



   .. py:method:: construct_adj(kg_graph)

      Get neighbors and corresponding relations for each entity in the KG.

      :param kg_graph: an undirected graph
      :type kg_graph: scipy.sparse.coo_matrix

      :returns:

                    - adj_entity(torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                      shape: [n_entities, neighbor_sample_size]
                    - adj_relation(torch.LongTensor): each line stores the corresponding sampled neighbor relations,
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



   .. py:method:: mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

      Mix neighbor vectors on user-specific graph.

      :param neighbor_vectors: The embeddings of neighbor entities(items),
                               shape: [batch_size, -1, neighbor_sample_size, embedding_size]
      :type neighbor_vectors: torch.FloatTensor
      :param neighbor_relations: The embeddings of neighbor relations,
                                 shape: [batch_size, -1, neighbor_sample_size, embedding_size]
      :type neighbor_relations: torch.FloatTensor
      :param user_embeddings: The embeddings of users, shape: [batch_size, embedding_size]
      :type user_embeddings: torch.FloatTensor

      :returns: The neighbors aggregated embeddings,
                shape: [batch_size, -1, embedding_size]
      :rtype: neighbors_aggregated(torch.FloatTensor)



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



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



