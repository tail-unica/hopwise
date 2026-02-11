hopwise.model.sequential_recommender.stamp
==========================================

.. py:module:: hopwise.model.sequential_recommender.stamp

.. autoapi-nested-parse::

   STAMP
   ################################################

   Reference:
       Qiao Liu et al. "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation." in KDD 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.stamp.STAMP


Module Contents
---------------

.. py:class:: STAMP(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   STAMP is capable of capturing users’ general interests from the long-term memory of a session context,
   whilst taking into account users’ current interests from the short-term memory of the last-clicks.


   .. note::

      According to the test results, we made a little modification to the score function mentioned in the paper,
      and did not use the final sigmoid activation function.


   .. py:attribute:: embedding_size


   .. py:attribute:: item_embedding


   .. py:attribute:: w1


   .. py:attribute:: w2


   .. py:attribute:: w3


   .. py:attribute:: w0


   .. py:attribute:: b_a


   .. py:attribute:: mlp_a


   .. py:attribute:: mlp_b


   .. py:attribute:: sigmoid


   .. py:attribute:: tanh


   .. py:attribute:: loss_type


   .. py:method:: _init_weights(module)


   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: count_alpha(context, aspect, output)

      This is a function that count the attention weights

      :param context: Item list embedding matrix, shape of [batch_size, time_steps, emb]
      :type context: torch.FloatTensor
      :param aspect: The embedding matrix of the last click item, shape of [batch_size, emb]
      :type aspect: torch.FloatTensor
      :param output: The average of the context, shape of [batch_size, emb]
      :type output: torch.FloatTensor

      :returns: attention weights, shape of [batch_size, time_steps]
      :rtype: torch.Tensor



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



