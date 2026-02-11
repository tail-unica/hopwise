hopwise.model.sequential_recommender.ksr
========================================

.. py:module:: hopwise.model.sequential_recommender.ksr

.. autoapi-nested-parse::

   KSR
   ################################################

   Reference:
       Jin Huang et al. "Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks."
       In SIGIR 2018



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.ksr.KSR


Module Contents
---------------

.. py:class:: KSR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   KSR integrates the RNN-based networks with Key-Value Memory Network (KV-MN).
   And it further incorporates knowledge base (KB) information to enhance the semantic representation of KV-MN.



   .. py:attribute:: ENTITY_ID


   .. py:attribute:: RELATION_ID


   .. py:attribute:: n_entities


   .. py:attribute:: n_relations


   .. py:attribute:: entity_embedding_matrix


   .. py:attribute:: relation_embedding_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: loss_type


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: gamma


   .. py:attribute:: device


   .. py:attribute:: freeze_kg


   .. py:attribute:: item_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: gru_layers


   .. py:attribute:: dense


   .. py:attribute:: dense_layer_u


   .. py:attribute:: dense_layer_i


   .. py:attribute:: relation_matrix


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _get_kg_embedding(head)

      Difference:
      We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems.



   .. py:method:: _memory_update_cell(user_memory, update_memory)


   .. py:method:: memory_update(item_seq, item_seq_len)

      Define write operator



   .. py:method:: memory_read(seq_output, user_memory)

      Define read operator



   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: _get_item_comb_embedding(item)


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



