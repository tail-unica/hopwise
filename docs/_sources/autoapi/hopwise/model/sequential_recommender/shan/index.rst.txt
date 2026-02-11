hopwise.model.sequential_recommender.shan
=========================================

.. py:module:: hopwise.model.sequential_recommender.shan

.. autoapi-nested-parse::

   SHAN
   ################################################

   Reference:
       Ying, H et al. "Sequential Recommender System based on Hierarchical Attention Network."in IJCAI 2018




Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.shan.SHAN


Module Contents
---------------

.. py:class:: SHAN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SHAN exploit the Hierarchical Attention Network to get the long-short term preference
   first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose



   .. py:attribute:: n_users


   .. py:attribute:: device


   .. py:attribute:: INVERSE_ITEM_SEQ


   .. py:attribute:: embedding_size


   .. py:attribute:: short_item_length


   .. py:attribute:: reg_weight


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: long_w


   .. py:attribute:: long_b


   .. py:attribute:: long_short_w


   .. py:attribute:: long_short_b


   .. py:attribute:: relu


   .. py:attribute:: loss_type


   .. py:method:: reg_loss(user_embedding, item_embedding)


   .. py:method:: init_weights(module)


   .. py:method:: forward(seq_item, user)


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



   .. py:method:: long_and_short_term_attention_based_pooling_layer(long_short_item_embedding, user_embedding, mask=None)

      Fusing the long term purpose with the short-term preference



   .. py:method:: long_term_attention_based_pooling_layer(seq_item_embedding, user_embedding, mask=None)

      Get the long term purpose of user



