hopwise.model.knowledge_aware_recommender.mkr
=============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.mkr

.. autoapi-nested-parse::

   MKR
   #####################################################
   Reference:
       Hongwei Wang et al. "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation." in WWW 2019.

   Reference code:
       https://github.com/hsientzucheng/MKR.PyTorch



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.mkr.MKR
   hopwise.model.knowledge_aware_recommender.mkr.CrossCompressUnit


Module Contents
---------------

.. py:class:: MKR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   MKR is a Multi-task feature learning approach for Knowledge graph enhanced Recommendation. It is a deep
   end-to-end framework that utilizes knowledge graph embedding task to assist recommendation task. The two
   tasks are associated by cross&compress units, which automatically share latent features and learn high-order
   interactions between items in recommender systems and entities in the knowledge graph.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: L


   .. py:attribute:: H


   .. py:attribute:: reg_weight


   .. py:attribute:: use_inner_product


   .. py:attribute:: dropout_prob


   .. py:attribute:: user_embeddings_lookup


   .. py:attribute:: item_embeddings_lookup


   .. py:attribute:: entity_embeddings_lookup


   .. py:attribute:: relation_embeddings_lookup


   .. py:attribute:: user_mlp


   .. py:attribute:: tail_mlp


   .. py:attribute:: cc_unit


   .. py:attribute:: kge_mlp


   .. py:attribute:: kge_pred_mlp


   .. py:attribute:: sigmoid_BCE


   .. py:method:: forward(user_indices=None, item_indices=None, head_indices=None, relation_indices=None, tail_indices=None)


   .. py:method:: _l2_loss(inputs)


   .. py:method:: calculate_rs_loss(interaction)

      Calculate the training loss for a batch data of RS.



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.



   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



.. py:class:: CrossCompressUnit(dim)

   Bases: :py:obj:`torch.nn.Module`


   This is Cross&Compress Unit for MKR model to model feature interactions between items and entities.


   .. py:attribute:: dim


   .. py:attribute:: fc_vv


   .. py:attribute:: fc_ev


   .. py:attribute:: fc_ve


   .. py:attribute:: fc_ee


   .. py:method:: forward(inputs)


