hopwise.model.sequential_recommender.gru4reckg
==============================================

.. py:module:: hopwise.model.sequential_recommender.gru4reckg

.. autoapi-nested-parse::

   GRU4RecKG
   ################################################



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.gru4reckg.GRU4RecKG


Module Contents
---------------

.. py:class:: GRU4RecKG(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   It is an extension of GRU4Rec, which concatenates item and its corresponding
   pre-trained knowledge graph embedding feature as the input.



   .. py:attribute:: entity_embedding_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: num_layers


   .. py:attribute:: dropout


   .. py:attribute:: freeze_kg


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: item_emb_dropout


   .. py:attribute:: entity_emb_dropout


   .. py:attribute:: item_gru_layers


   .. py:attribute:: entity_gru_layers


   .. py:attribute:: dense_layer


   .. py:method:: forward(item_seq, item_seq_len)


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



