hopwise.model.sequential_recommender.dien
=========================================

.. py:module:: hopwise.model.sequential_recommender.dien

.. autoapi-nested-parse::

   DIEN
   ##############################################
   Reference:
       Guorui Zhou et al. "Deep Interest Evolution Network for Click-Through Rate Prediction" in AAAI 2019

   Reference code:
       - https://github.com/mouna99/dien
       - https://github.com/shenweichen/DeepCTR-Torch/



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.dien.DIEN
   hopwise.model.sequential_recommender.dien.InterestExtractorNetwork
   hopwise.model.sequential_recommender.dien.InterestEvolvingLayer
   hopwise.model.sequential_recommender.dien.AGRUCell
   hopwise.model.sequential_recommender.dien.AUGRUCell
   hopwise.model.sequential_recommender.dien.DynamicRNN


Module Contents
---------------

.. py:class:: DIEN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   DIEN has an interest extractor layer to capture temporal interests from history behavior sequence,and an
   interest evolving layer to capture interest evolving process that is relative to the target item. At interest
   evolving layer, attention mechanism is embedded intothe sequential structure novelly, and the effects of relative
   interests are strengthened during interest evolution.



   .. py:attribute:: input_type


   .. py:attribute:: device


   .. py:attribute:: alpha


   .. py:attribute:: gru


   .. py:attribute:: pooling_mode


   .. py:attribute:: dropout_prob


   .. py:attribute:: LABEL_FIELD


   .. py:attribute:: embedding_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: NEG_ITEM_SEQ


   .. py:attribute:: types
      :value: ['user', 'item']



   .. py:attribute:: user_feat


   .. py:attribute:: item_feat


   .. py:attribute:: att_list


   .. py:attribute:: interest_mlp_list


   .. py:attribute:: dnn_mlp_list


   .. py:attribute:: interset_extractor


   .. py:attribute:: interest_evolution


   .. py:attribute:: embedding_layer


   .. py:attribute:: dnn_mlp_layers


   .. py:attribute:: dnn_predict_layer


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:attribute:: other_parameter_name
      :value: ['embedding_layer']



   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item_seq, neg_item_seq, item_seq_len, next_items)


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



.. py:class:: InterestExtractorNetwork(input_size, hidden_size, mlp_size)

   Bases: :py:obj:`torch.nn.Module`


   In e-commerce system, user behavior is the carrier of latent interest, and interest will change after
   user takes one behavior. At the interest extractor layer, DIEN extracts series of interest states from
   sequential user behaviors.


   .. py:attribute:: gru


   .. py:attribute:: auxiliary_net


   .. py:method:: forward(keys, keys_length, neg_keys=None)


   .. py:method:: auxiliary_loss(h_states, click_seq, noclick_seq, keys_length)

      Computes the auxiliary loss.

      :param h_states: The output of GRUs' hidden layer,
                       shape [batch_size, history_length - 1, embedding_size].
      :type h_states: torch.Tensor
      :param click_seq: The sequence that users consumed,
                        shape [batch_size, history_length - 1, embedding_size].
      :type click_seq: torch.Tensor
      :param noclick_seq: The sequence that users did not consume,
                          shape [batch_size, history_length - 1, embedding_size].
      :type noclick_seq: torch.Tensor
      :param keys_length: The true length of the user history sequence.
      :type keys_length: torch.Tensor

      :returns: auxiliary loss
      :rtype: torch.Tensor



.. py:class:: InterestEvolvingLayer(mask_mat, input_size, rnn_hidden_size, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=True, gru='GRU')

   Bases: :py:obj:`torch.nn.Module`


   As the joint influence from external environment and internal cognition, different kinds of user interests are
   evolving over time. Interest Evolving Layer can capture interest evolving process that is relative to the target
   item.


   .. py:attribute:: mask_mat


   .. py:attribute:: gru
      :value: 'GRU'



   .. py:method:: final_output(outputs, keys_length)

      Get the last effective value in the interest evolution sequence
      :param outputs: the output of `DynamicRNN` after `pad_packed_sequence`
      :type outputs: torch.Tensor
      :param keys_length: the true length of the user history sequence
      :type keys_length: torch.Tensor

      :returns: The user's CTR for the next item
      :rtype: torch.Tensor



   .. py:method:: forward(queries, keys, keys_length)


.. py:class:: AGRUCell(input_size, hidden_size, bias=True)

   Bases: :py:obj:`torch.nn.Module`


   Attention based GRU (AGRU). AGRU uses the attention score to replace the update gate of GRU, and changes the
       hidden state directly.

       Formally:
           ..math: {h}_{t}^{\prime}=\left(1-a_{t}
   ight) * {h}_{t-1}^{\prime}+a_{t} *      ilde{{h}}_{t}^{\prime}

           :math:`{h}_{t}^{\prime}`, :math:`h_{t-1}^{\prime}`, :math:`{h}_{t-1}^{\prime}`,
           :math: `        ilde{{h}}_{t}^{\prime}` are the hidden state of AGRU




   .. py:attribute:: input_size


   .. py:attribute:: hidden_size


   .. py:attribute:: bias
      :value: True



   .. py:attribute:: weight_ih


   .. py:attribute:: weight_hh


   .. py:method:: forward(input, hidden_output, att_score)


.. py:class:: AUGRUCell(input_size, hidden_size, bias=True)

   Bases: :py:obj:`torch.nn.Module`


   Effect of GRU with attentional update gate (AUGRU). AUGRU combines attention mechanism and GRU seamlessly.

       Formally:
           ..math:         ilde{{u}}_{t}^{\prime}=a_{t} * {u}_{t}^{\prime} \
                   {h}_{t}^{\prime}=\left(1-       ilde{{u}}_{t}^{\prime}
   ight) \circ {h}_{t-1}^{\prime}+ ilde{{u}}_{t}^{\prime} \circ    ilde{{h}}_{t}^{\prime}




   .. py:attribute:: input_size


   .. py:attribute:: hidden_size


   .. py:attribute:: bias
      :value: True



   .. py:attribute:: weight_ih


   .. py:attribute:: weight_hh


   .. py:method:: forward(input, hidden_output, att_score)


.. py:class:: DynamicRNN(input_size, hidden_size, bias=True, gru='AGRU')

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: input_size


   .. py:attribute:: hidden_size


   .. py:method:: forward(input, att_scores=None, hidden_output=None)


