hopwise.model.knowledge_aware_recommender.cfkg
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.cfkg

.. autoapi-nested-parse::

   CFKG
   ##################################################
   Reference:
       Qingyao Ai et al. "Learning heterogeneous knowledge base embeddings for explainable recommendation." in MDPI 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.cfkg.CFKG


Module Contents
---------------

.. py:class:: CFKG(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   CFKG is a knowledge-based recommendation model, it combines knowledge graph and the user-item interaction
   graph to a new graph. In this graph, user, item and related attribute are viewed as entities, and the interaction
   between user and item and the link between item and attribute are viewed as relations. It define a new score
   function as follows:

   .. math::
       d (u_i + r_{buy}, v_j)

   .. note::

      In the original paper, CFKG puts recommender data (u-i interaction) and knowledge data (h-r-t) together
      for sampling and mix them for training. In this version, we sample recommender data
      and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: rec_loss


   .. py:method:: forward(user, item)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: _get_score(h_e, t_e, r_e)


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



