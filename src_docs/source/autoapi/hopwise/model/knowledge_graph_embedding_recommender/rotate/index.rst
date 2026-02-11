hopwise.model.knowledge_graph_embedding_recommender.rotate
==========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.rotate

.. autoapi-nested-parse::

   RotatE
   ##################################################
   Reference:
       Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." in ICLR 2019.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.rotate.RotatE


Module Contents
---------------

.. py:class:: RotatE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   RotatE models relations as rotations in a complex latent space with h, r, t belonging
   to the set of d-dimensional complex numbers. The embedding for r belonging to the set of d-dimensional
   complex numbers, is a rotation vector: in all its elements, the phase conveys the rotation along that axis,
   and the modulus is equal to 1.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: user_embedding_im


   .. py:attribute:: entity_embedding


   .. py:attribute:: entity_embedding_im


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head_re, head_im, relation, tail_re, tail_im)


   .. py:method:: _get_rec_embeddings(user, positive_items, negative_items)


   .. py:method:: _get_kg_embeddings(head, relation, positive_tails, negative_tails)


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



