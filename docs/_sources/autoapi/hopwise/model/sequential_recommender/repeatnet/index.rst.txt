hopwise.model.sequential_recommender.repeatnet
==============================================

.. py:module:: hopwise.model.sequential_recommender.repeatnet

.. autoapi-nested-parse::

   RepeatNet
   ################################################

   Reference:
       Pengjie Ren et al. "RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation."
       in AAAI 2019

   Reference code:
       https://github.com/PengjieRen/RepeatNet.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.repeatnet.RepeatNet
   hopwise.model.sequential_recommender.repeatnet.Repeat_Explore_Mechanism
   hopwise.model.sequential_recommender.repeatnet.Repeat_Recommendation_Decoder
   hopwise.model.sequential_recommender.repeatnet.Explore_Recommendation_Decoder


Module Contents
---------------

.. py:class:: RepeatNet(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   RepeatNet explores a hybrid encoder with an repeat module and explore module
   repeat module is used for finding out the repeat consume in sequential recommendation
   explore module is used for exploring new items for recommendation



   .. py:attribute:: input_type


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: joint_train


   .. py:attribute:: dropout_prob


   .. py:attribute:: item_matrix


   .. py:attribute:: gru


   .. py:attribute:: repeat_explore_mechanism


   .. py:attribute:: repeat_recommendation_decoder


   .. py:attribute:: explore_recommendation_decoder


   .. py:attribute:: loss_fct


   .. py:method:: _init_weights(module)


   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: repeat_explore_loss(item_seq, pos_item)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



.. py:class:: Repeat_Explore_Mechanism(device, hidden_size, seq_len, dropout_prob)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: dropout


   .. py:attribute:: hidden_size


   .. py:attribute:: device


   .. py:attribute:: seq_len


   .. py:attribute:: Wre


   .. py:attribute:: Ure


   .. py:attribute:: tanh


   .. py:attribute:: Vre


   .. py:attribute:: Wcre


   .. py:method:: forward(all_memory, last_memory)

      Calculate the probability of Repeat and explore



.. py:class:: Repeat_Recommendation_Decoder(device, hidden_size, seq_len, num_item, dropout_prob)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: dropout


   .. py:attribute:: hidden_size


   .. py:attribute:: device


   .. py:attribute:: seq_len


   .. py:attribute:: num_item


   .. py:attribute:: Wr


   .. py:attribute:: Ur


   .. py:attribute:: tanh


   .. py:attribute:: Vr


   .. py:method:: forward(all_memory, last_memory, item_seq, mask=None)

      Calculate the the force of repeat



.. py:class:: Explore_Recommendation_Decoder(hidden_size, seq_len, num_item, device, dropout_prob)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: dropout


   .. py:attribute:: hidden_size


   .. py:attribute:: seq_len


   .. py:attribute:: num_item


   .. py:attribute:: device


   .. py:attribute:: We


   .. py:attribute:: Ue


   .. py:attribute:: tanh


   .. py:attribute:: Ve


   .. py:attribute:: matrix_for_explore


   .. py:method:: forward(all_memory, last_memory, item_seq, mask=None)

      Calculate the force of explore



