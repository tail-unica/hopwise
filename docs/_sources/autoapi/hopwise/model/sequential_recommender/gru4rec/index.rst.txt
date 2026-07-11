hopwise.model.sequential_recommender.gru4rec
============================================

.. py:module:: hopwise.model.sequential_recommender.gru4rec

.. autoapi-nested-parse::

   GRU4Rec
   ################################################

   Reference:
       Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.gru4rec.GRU4Rec


Module Contents
---------------

.. py:class:: GRU4Rec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   GRU4Rec is a model that incorporate RNN for recommendation.

   .. note::

      Regarding the innovation of this article,we can only achieve the data augmentation mentioned
      in the paper and directly output the embedding of the item,
      in order that the generation method we used is common to other sequential models.


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: loss_type


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: item_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: gru_layers


   .. py:attribute:: dense


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



