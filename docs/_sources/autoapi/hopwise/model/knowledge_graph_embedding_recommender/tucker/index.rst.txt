hopwise.model.knowledge_graph_embedding_recommender.tucker
==========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.tucker

.. autoapi-nested-parse::

   TuckER
   ##################################################
   Reference:
       Balažević et al. "TuckER: Tensor Factorization for Knowledge Graph Completion." in EMNLP/IJCNLP 2019.

   Reference code:
       https://github.com/ibalazevic/TuckER



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.tucker.TuckER


Module Contents
---------------

.. py:class:: TuckER(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TuckER relies on Tucker Decomposition. It handles entity and relation embeddings of independent dimension
   and jointly learns a share core W.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: label_smoothing


   .. py:attribute:: input_dropout


   .. py:attribute:: input_dropout1


   .. py:attribute:: input_dropout2


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: weights


   .. py:attribute:: hidden_dropout1


   .. py:attribute:: hidden_dropout2


   .. py:attribute:: bn0


   .. py:attribute:: bn1


   .. py:attribute:: loss


   .. py:method:: forward(h, r, embeddings)


   .. py:method:: _get_rec_embeddings(user)


   .. py:method:: _get_kg_embeddings(h, r)


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



