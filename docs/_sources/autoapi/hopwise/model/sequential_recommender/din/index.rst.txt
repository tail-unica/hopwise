hopwise.model.sequential_recommender.din
========================================

.. py:module:: hopwise.model.sequential_recommender.din

.. autoapi-nested-parse::

   DIN
   ##############################################
   Reference:
       Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

   Reference code:
       - https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din
       - https://github.com/shenweichen/DeepCTR-Torch/tree/master/deepctr_torch/models



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.din.DIN


Module Contents
---------------

.. py:class:: DIN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   Deep Interest Network utilizes the attention mechanism to get the weight of each user's behavior according
   to the target items, and finally gets the user representation.

   .. note::

      In the official source code, unlike the paper, user features and context features are not input into DNN.
      We just migrated and changed the official source code.
      But You can get user features embedding from user_feat_list.
      Besides, in order to compare with other models, we use AUC instead of GAUC to evaluate the model.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL_FIELD


   .. py:attribute:: embedding_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: device


   .. py:attribute:: pooling_mode


   .. py:attribute:: dropout_prob


   .. py:attribute:: types
      :value: ['user', 'item']



   .. py:attribute:: user_feat


   .. py:attribute:: item_feat


   .. py:attribute:: dnn_list


   .. py:attribute:: att_list


   .. py:attribute:: attention


   .. py:attribute:: dnn_mlp_layers


   .. py:attribute:: embedding_layer


   .. py:attribute:: dnn_predict_layers


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:attribute:: other_parameter_name
      :value: ['embedding_layer']



   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item_seq, item_seq_len, next_items)


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



