hopwise.model.knowledge_aware_recommender.mcclk
===============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.mcclk

.. autoapi-nested-parse::

   MCCLK
   ##################################################
   Reference:
       Ding Zou et al. "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." in SIGIR 2022.

   Reference code:
       https://github.com/CCIIPLab/MCCLK



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.mcclk.Aggregator
   hopwise.model.knowledge_aware_recommender.mcclk.GraphConv
   hopwise.model.knowledge_aware_recommender.mcclk.MCCLK


Module Contents
---------------

.. py:class:: Aggregator(item_only=False, attention=True)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: item_only
      :value: False



   .. py:attribute:: attention
      :value: True



   .. py:method:: forward(entity_emb, user_emb, relation_emb, edge_index, edge_type, inter_matrix)


   .. py:method:: calculate_sim_hrt(entity_emb_head, entity_emb_tail, relation_emb)

      The calculation method of attention weight here follows the code implementation of the author, which is
      slightly different from that described in the paper.



.. py:class:: GraphConv(config, embedding_size, n_relations, edge_index, edge_type, inter_matrix, device)

   Bases: :py:obj:`torch.nn.Module`


   Graph Convolutional Network


   .. py:attribute:: n_relations


   .. py:attribute:: edge_index


   .. py:attribute:: edge_type


   .. py:attribute:: inter_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: n_hops


   .. py:attribute:: node_dropout_rate


   .. py:attribute:: mess_dropout_rate


   .. py:attribute:: topk


   .. py:attribute:: lambda_coeff


   .. py:attribute:: build_graph_separately


   .. py:attribute:: device


   .. py:attribute:: relation_embedding


   .. py:attribute:: convs


   .. py:attribute:: node_dropout


   .. py:attribute:: mess_dropout


   .. py:method:: edge_sampling(edge_index, edge_type, rate=0.5)


   .. py:method:: forward(user_emb, entity_emb)


   .. py:method:: build_adj(context, topk)

      Construct a k-Nearest-Neighbor item-item semantic graph.

      :returns: Sparse tensor of the normalized item-item matrix.



   .. py:method:: _build_graph_separately(entity_emb)


.. py:class:: MCCLK(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   MCCLK is a knowledge-based recommendation model.
   It focuses on the contrastive learning in KG-aware recommendation and proposes a novel multi-level cross-view
   contrastive learning mechanism. This model comprehensively considers three different graph views for KG-aware
   recommendation, including global-level structural view, local-level collaborative and semantic views. It hence
   performs contrastive learning across three views on both local and global levels, mining comprehensive graph
   feature and structure information in a self-supervised manner.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: lightgcn_layer


   .. py:attribute:: item_agg_layer


   .. py:attribute:: temperature


   .. py:attribute:: alpha


   .. py:attribute:: beta


   .. py:attribute:: loss_type


   .. py:attribute:: inter_matrix


   .. py:attribute:: inter_graph


   .. py:attribute:: kg_graph


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: gcn


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: light_gcn(user_embedding, item_embedding, adj)


   .. py:method:: sim(z1: torch.Tensor, z2: torch.Tensor)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: local_level_loss(A_embedding, B_embedding)


   .. py:method:: global_level_loss_1(A_embedding, B_embedding)


   .. py:method:: global_level_loss_2(A_embedding, B_embedding)


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



