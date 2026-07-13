hopwise.model.sequential_recommender.sine
=========================================

.. py:module:: hopwise.model.sequential_recommender.sine

.. autoapi-nested-parse::

   SINE
   ################################################

   Reference:
       Qiaoyu Tan et al. "Sparse-Interest Network for Sequential Recommendation." in WSDM 2021.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.sine.SINE


Module Contents
---------------

.. py:class:: SINE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: input_type


   .. py:attribute:: n_users


   .. py:attribute:: n_items


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: D


   .. py:attribute:: L


   .. py:attribute:: k


   .. py:attribute:: tau


   .. py:attribute:: reg_loss_ratio


   .. py:attribute:: initializer_range
      :value: 0.01



   .. py:attribute:: w1


   .. py:attribute:: w2


   .. py:attribute:: w3


   .. py:attribute:: w4


   .. py:attribute:: C


   .. py:attribute:: w_k_1


   .. py:attribute:: w_k_2


   .. py:attribute:: item_embedding


   .. py:attribute:: ln2


   .. py:attribute:: ln4


   .. py:method:: _init_weight(shape)


   .. py:method:: _init_weights(module)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_reg_loss()


   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



