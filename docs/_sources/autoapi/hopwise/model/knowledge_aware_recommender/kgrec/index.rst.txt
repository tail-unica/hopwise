hopwise.model.knowledge_aware_recommender.kgrec
===============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kgrec

.. autoapi-nested-parse::

   KGREC
   ##################################################
   Reference:
       Yuhao Yang et al. "Knowledge Graph Self-Supervised Rationalization for Recommendation" in WWW 2021.
   Reference code:
       https://github.com/HKUDS/KGRec



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kgrec.Contrast
   hopwise.model.knowledge_aware_recommender.kgrec.AttnHGCN
   hopwise.model.knowledge_aware_recommender.kgrec.KGRec


Module Contents
---------------

.. py:class:: Contrast(num_hidden: int, tau: float = 0.7)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: tau
      :type:  float
      :value: 0.7



   .. py:attribute:: mlp1


   .. py:attribute:: mlp2


   .. py:method:: sim(z1: torch.Tensor, z2: torch.Tensor)


   .. py:method:: self_sim(z1, z2)


   .. py:method:: loss(z1: torch.Tensor, z2: torch.Tensor)


   .. py:method:: forward(z1: torch.Tensor, z2: torch.Tensor)


.. py:class:: AttnHGCN(embedding_size, n_hops, n_users, n_relations, mess_dropout_rate=0.1)

   Bases: :py:obj:`torch.nn.Module`


   Heterogeneous Graph Convolutional Network


   .. py:attribute:: no_attn_convs


   .. py:attribute:: embedding_size


   .. py:attribute:: n_hops


   .. py:attribute:: n_relations


   .. py:attribute:: n_users


   .. py:attribute:: mess_dropout_rate
      :value: 0.1



   .. py:attribute:: relation_embedding


   .. py:attribute:: W_Q


   .. py:attribute:: n_heads
      :value: 2



   .. py:attribute:: d_k


   .. py:attribute:: mess_dropout


   .. py:method:: shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w)


   .. py:method:: forward(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, item_attn=None)


   .. py:method:: forward_ui(user_emb, item_emb, inter_edge, inter_edge_w)


   .. py:method:: forward_kg(entity_emb, edge_index, edge_type)


   .. py:method:: ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)


   .. py:method:: kg_agg(entity_emb, edge_index, edge_type)


   .. py:method:: norm_attn_computer(entity_emb, edge_index, edge_type=None, return_logits=False)


.. py:class:: KGRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGRec is a self-supervised knowledge-aware recommender that identifies and focuses on informative knowledge
   graph connections through an attentive rationalization mechanism. It combines generative masking reconstruction
   and contrastive learning tasks to highlight and align meaningful knowledge and interaction signals. By masking
   and rebuilding high-rationale edges while filtering noisy ones, KGRec learns more interpretable and noise-resistant
   recommendations.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: context_hops


   .. py:attribute:: node_dropout_rate


   .. py:attribute:: mess_dropout_rate


   .. py:attribute:: mae_coef


   .. py:attribute:: mae_msize


   .. py:attribute:: cl_coef


   .. py:attribute:: cl_tau


   .. py:attribute:: cl_drop


   .. py:attribute:: samp_func


   .. py:attribute:: inter_edge


   .. py:attribute:: kg_graph


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:attribute:: gcn


   .. py:attribute:: contrast_fn


   .. py:attribute:: node_dropout


   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data of KG.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: relation_aware_edge_sampling(sampling_rate=0.5)


   .. py:method:: edge_sampling(edge_index, edge_type, sampling_rate=0.5)


   .. py:method:: mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id)


   .. py:method:: adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score)


   .. py:method:: adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w)


   .. py:method:: create_mae_loss(node_pair_emb, masked_edge_emb=None)


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



