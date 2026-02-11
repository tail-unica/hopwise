hopwise.model.knowledge_graph_embedding_recommender.conve
=========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.conve

.. autoapi-nested-parse::

   ConvE
   ##################################################
   Reference:
       Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings." in AAAI 2018.

   Reference code:
       https://github.com/TimDettmers/ConvE



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.conve.ConvE


Module Contents
---------------

.. py:class:: ConvE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   ConvE represent h,r,t in a subset of real number in d dimension. When scoring them,
   it concatenates and reshape h and r into a unique input [h;r]. This input is passed through
   a convolutional layers with a set of k filters and then through a dense layer with d neurons
   and a set of weight W. The output is finally combined with the tail embedding t
   using the dot product to produce the final score.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: label_smoothing


   .. py:attribute:: input_dropout


   .. py:attribute:: hidden_dropout


   .. py:attribute:: feature_dropout


   .. py:attribute:: embedding_dim1


   .. py:attribute:: embedding_dim2


   .. py:attribute:: hidden_size


   .. py:attribute:: use_bias


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relations_embeddings


   .. py:attribute:: inp_drop


   .. py:attribute:: hidden_drop


   .. py:attribute:: feature_map_drop


   .. py:attribute:: conv1


   .. py:attribute:: bn0


   .. py:attribute:: bn1


   .. py:attribute:: bn2


   .. py:attribute:: fc


   .. py:attribute:: loss


   .. py:method:: forward(head, relation, embeddings, bias)


   .. py:method:: _get_rec_embeddings(user)


   .. py:method:: _get_kg_embeddings(head, relation)


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



