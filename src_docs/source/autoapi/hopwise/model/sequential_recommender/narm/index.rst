hopwise.model.sequential_recommender.narm
=========================================

.. py:module:: hopwise.model.sequential_recommender.narm

.. autoapi-nested-parse::

   NARM
   ################################################

   Reference:
       Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

   Reference code:
       https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.narm.NARM


Module Contents
---------------

.. py:class:: NARM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
   and capture the user’s main purpose in the current session.



   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: n_layers


   .. py:attribute:: dropout_probs


   .. py:attribute:: device


   .. py:attribute:: item_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: gru


   .. py:attribute:: a_1


   .. py:attribute:: a_2


   .. py:attribute:: v_t


   .. py:attribute:: ct_dropout


   .. py:attribute:: b


   .. py:attribute:: loss_type


   .. py:method:: _init_weights(module)


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



