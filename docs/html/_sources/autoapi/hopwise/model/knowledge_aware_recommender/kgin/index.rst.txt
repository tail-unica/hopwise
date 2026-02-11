hopwise.model.knowledge_aware_recommender.kgin
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kgin

.. autoapi-nested-parse::

   KGIN
   ##################################################
   Reference:
       Xiang Wang et al. "Learning Intents behind Interactions with Knowledge Graph for Recommendation." in WWW 2021.
   Reference code:
       https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kgin.Aggregator
   hopwise.model.knowledge_aware_recommender.kgin.GraphConv
   hopwise.model.knowledge_aware_recommender.kgin.KGIN


Module Contents
---------------

.. py:class:: Aggregator

   Bases: :py:obj:`torch.nn.Module`


   Relational Path-aware Convolution Network


   .. py:method:: forward(entity_emb, user_emb, latent_emb, relation_emb, edge_index, edge_type, interact_mat, disen_weight_att)


.. py:class:: GraphConv(embedding_size, n_hops, n_users, n_factors, n_relations, edge_index, edge_type, interact_mat, ind, tmp, device, node_dropout_rate=0.5, mess_dropout_rate=0.1)

   Bases: :py:obj:`torch.nn.Module`


   Graph Convolutional Network


   .. py:attribute:: embedding_size


   .. py:attribute:: n_hops


   .. py:attribute:: n_relations


   .. py:attribute:: n_users


   .. py:attribute:: n_factors


   .. py:attribute:: edge_index


   .. py:attribute:: edge_type


   .. py:attribute:: interact_mat


   .. py:attribute:: node_dropout_rate
      :value: 0.5



   .. py:attribute:: mess_dropout_rate
      :value: 0.1



   .. py:attribute:: ind


   .. py:attribute:: temperature


   .. py:attribute:: device


   .. py:attribute:: relation_embedding


   .. py:attribute:: disen_weight_att


   .. py:attribute:: convs


   .. py:attribute:: node_dropout


   .. py:attribute:: mess_dropout


   .. py:method:: edge_sampling(edge_index, edge_type, rate=0.5)


   .. py:method:: forward(user_emb, entity_emb, latent_emb)

      Node dropout



   .. py:method:: calculate_cor_loss(tensors)


.. py:class:: KGIN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGIN is a knowledge-aware recommendation model. It combines knowledge graph and the user-item interaction
   graph to a new graph called collaborative knowledge graph (CKG). This model explores intents behind a user-item
   interaction by using auxiliary item knowledge.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_factors


   .. py:attribute:: context_hops


   .. py:attribute:: node_dropout_rate


   .. py:attribute:: mess_dropout_rate


   .. py:attribute:: ind


   .. py:attribute:: sim_decay


   .. py:attribute:: reg_weight


   .. py:attribute:: temperature


   .. py:attribute:: interact_mat


   .. py:attribute:: kg_graph


   .. py:attribute:: n_nodes


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: latent_embedding


   .. py:attribute:: gcn


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data of KG.

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



