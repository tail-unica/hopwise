hopwise.model.knowledge_graph_embedding_recommender.analogy
===========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.analogy

.. autoapi-nested-parse::

   Analogy
   ##################################################
   Reference:
       Liu et al. "Analogical Inference for Multi-Relational Embeddings." in ICML 2017.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.analogy.Analogy


Module Contents
---------------

.. py:class:: Analogy(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   Analogy extends RESCAL so as to further model the analogical properties of entities and relations e.g.
   Interstellar is to Fantasy as Nolan is to Oppenheimer”.
   It employs the same scoring function as RESCAL but with some constraints.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: scalar_share


   .. py:attribute:: ui_relation


   .. py:attribute:: scalar_dim


   .. py:attribute:: complex_dim


   .. py:attribute:: user_embedding


   .. py:attribute:: user_re_embedding


   .. py:attribute:: user_im_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: entity_re_embedding


   .. py:attribute:: entity_im_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: relation_re_embedding


   .. py:attribute:: relation_im_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, tail_e, tail_re_e, tail_im_e)


   .. py:method:: _get_rec_embeddings(users, pos_items, neg_items)


   .. py:method:: _get_kg_embeddings(heads, relations, pos_tails, neg_tails)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



