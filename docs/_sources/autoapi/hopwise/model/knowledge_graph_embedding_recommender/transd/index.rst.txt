hopwise.model.knowledge_graph_embedding_recommender.transd
==========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.transd

.. autoapi-nested-parse::

   TransD
   ##################################################
   Reference:
       Ji et al. "Knowledge Graph Embedding via Dynamic Mapping Matrix." in ACL/IJCNLP 2015.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.transd.TransD


Module Contents
---------------

.. py:class:: TransD(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransD simplifies TransR by further decomposing the projection matrix into a product of two vector.
   Also in this case, the scoring function is the same as TransH and TransR,
   but it introduces three additional mapping vectors along with the entity and relation representation.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: user_vec_embedding


   .. py:attribute:: entity_vec_embedding


   .. py:attribute:: relation_vec_embedding


   .. py:attribute:: loss


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_rec_vec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: _get_kg_vec_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: forward(ent, ent_vect, rel_vect)

      We note that :math:`p_r(e)_i = e^p^Te \times r^p_i + e_i` which is
      more efficient to compute than the matrix formulation in the original
      paper.



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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



