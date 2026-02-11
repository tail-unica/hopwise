hopwise.model.knowledge_graph_embedding_recommender.hole
========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.hole

.. autoapi-nested-parse::

   HolE
   ##################################################
   Reference:
       Nickel et al. "Holographic embeddings of knowledge graphs." in AAAI 2016.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.hole.HolE


Module Contents
---------------

.. py:class:: HolE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   HoLE combines the expressive power of RESCAL with the efficiency and simplicity of DistMult.
   The entity representations are composed into h ⋆ t in the set of real numbers,
   with the circular correlation operator.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: forward(h, r, t)


   .. py:method:: get_rolling_matrix(x)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


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



