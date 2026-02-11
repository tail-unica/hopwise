hopwise.model.sequential_recommender.core
=========================================

.. py:module:: hopwise.model.sequential_recommender.core

.. autoapi-nested-parse::

   CORE
   ################################################
   Reference:
       Yupeng Hou, Binbin Hu, Zhiqiang Zhang, Wayne Xin Zhao. "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." in SIGIR 2022.

       https://github.com/RUCAIBox/CORE



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.core.TransNet
   hopwise.model.sequential_recommender.core.CORE


Module Contents
---------------

.. py:class:: TransNet(config, dataset)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: fn


   .. py:method:: get_attention_mask(item_seq, bidirectional=False)

      Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention.



   .. py:method:: forward(item_seq, item_emb)


   .. py:method:: _init_weights(module)

      Initialize the weights



.. py:class:: CORE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   CORE is a simple and effective framewor, which unifies the representation spac
   for both the encoding and decoding processes in session-based recommendation.


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: dnn_type


   .. py:attribute:: sess_dropout


   .. py:attribute:: item_dropout


   .. py:attribute:: temperature


   .. py:attribute:: item_embedding


   .. py:method:: _reset_parameters()


   .. py:method:: ave_net(item_seq, item_emb)
      :staticmethod:



   .. py:method:: forward(item_seq)


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



