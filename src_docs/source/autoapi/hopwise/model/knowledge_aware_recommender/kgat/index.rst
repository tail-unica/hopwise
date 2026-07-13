hopwise.model.knowledge_aware_recommender.kgat
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kgat

.. autoapi-nested-parse::

   KGAT
   ##################################################
   Reference:
       Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

   Reference code:
       https://github.com/xiangwang1223/knowledge_graph_attention_network



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kgat.Aggregator
   hopwise.model.knowledge_aware_recommender.kgat.KGAT


Module Contents
---------------

.. py:class:: Aggregator(input_dim, output_dim, dropout, aggregator_type)

   Bases: :py:obj:`torch.nn.Module`


   GNN Aggregator layer


   .. py:attribute:: input_dim


   .. py:attribute:: output_dim


   .. py:attribute:: dropout


   .. py:attribute:: aggregator_type


   .. py:attribute:: message_dropout


   .. py:attribute:: activation


   .. py:method:: forward(norm_matrix, ego_embeddings)


.. py:class:: KGAT(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
   graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
   items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.


   .. py:attribute:: input_type


   .. py:attribute:: ckg


   .. py:attribute:: all_hs


   .. py:attribute:: all_ts


   .. py:attribute:: all_rs


   .. py:attribute:: matrix_size


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: layers


   .. py:attribute:: aggregator_type


   .. py:attribute:: mess_dropout


   .. py:attribute:: reg_weight


   .. py:attribute:: A_in


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: trans_w


   .. py:attribute:: aggregator_layers


   .. py:attribute:: tanh


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_entity_e']



   .. py:method:: init_graph()

      Get the initial attention matrix through the collaborative knowledge graph

      :returns: Sparse tensor of the attention matrix
      :rtype: torch.sparse.FloatTensor



   .. py:method:: _get_ego_embeddings()


   .. py:method:: forward()


   .. py:method:: _get_kg_embedding(h, r, pos_t, neg_t)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: generate_transE_score(hs, ts, r)

      Calculating scores for triples in KG.

      :param hs: head entities
      :type hs: torch.Tensor
      :param ts: tail entities
      :type ts: torch.Tensor
      :param r: the relation id between hs and ts
      :type r: int

      :returns: the scores of (hs, r, ts)
      :rtype: torch.Tensor



   .. py:method:: update_attentive_A()

      Update the attention matrix using the updated embedding matrix



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



