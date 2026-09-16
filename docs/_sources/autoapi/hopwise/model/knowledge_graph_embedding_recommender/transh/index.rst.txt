hopwise.model.knowledge_graph_embedding_recommender.transh
==========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.transh

.. autoapi-nested-parse::

   TransH
   ##################################################
   Reference:
       Wang Z. et al. "Knowledge Graph Embedding by Translating on Hyperplanes." in AAAI 2014.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.transh.TransH


Module Contents
---------------

.. py:class:: TransH(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransH Have been invented to overcome the disadvantages of TransE,
   allowing an entity to have distinct representations when involved in different relations.
   It introduces relation-specific hyperplanes.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: norm_vec


   .. py:attribute:: rec_loss


   .. py:method:: forward(head, relation, tail, relation_ids)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: project(ent, rel)


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



