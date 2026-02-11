hopwise.model.general_recommender.dmf
=====================================

.. py:module:: hopwise.model.general_recommender.dmf

.. autoapi-nested-parse::

   DMF
   ################################################
   Reference:
       Hong-Jian Xue et al. "Deep Matrix Factorization Models for Recommender Systems." in IJCAI 2017.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.dmf.DMF


Module Contents
---------------

.. py:class:: DMF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   DMF is an neural network enhanced matrix factorization model.
   The original interaction matrix of :math:`n_{users} \times n_{items}` is set as model input,
   we carefully design the data interface and use sparse tensor to train and test efficiently.
   We just implement the model following the original author with a pointwise training mode.

   .. note::

      Our implementation is a improved version which is different from the original paper.
      For a better performance and stability, we replace cosine similarity to inner-product when calculate
      final score of user's and item's embedding.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: RATING


   .. py:attribute:: user_embedding_size


   .. py:attribute:: item_embedding_size


   .. py:attribute:: user_hidden_size_list


   .. py:attribute:: item_hidden_size_list


   .. py:attribute:: inter_matrix_type


   .. py:attribute:: max_rating


   .. py:attribute:: history_user_id


   .. py:attribute:: history_user_value


   .. py:attribute:: history_item_id


   .. py:attribute:: history_item_value


   .. py:attribute:: user_linear


   .. py:attribute:: item_linear


   .. py:attribute:: user_fc_layers


   .. py:attribute:: item_fc_layers


   .. py:attribute:: sigmoid


   .. py:attribute:: bce_loss


   .. py:attribute:: i_embedding
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['i_embedding']



   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item)


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



   .. py:method:: get_user_embedding(user)

      Get a batch of user's embedding with the user's id and history interaction matrix.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor

      :returns: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: get_item_embedding()

      Get all item's embedding with history interaction matrix.

      Considering the RAM of device, we use matrix multiply on sparse tensor for generalization.

      :returns: The embedding tensor of all item, shape: [n_items, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



