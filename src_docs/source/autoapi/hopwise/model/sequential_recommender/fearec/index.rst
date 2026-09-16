hopwise.model.sequential_recommender.fearec
===========================================

.. py:module:: hopwise.model.sequential_recommender.fearec

.. autoapi-nested-parse::

   FEARec
   ################################################

   Reference:
       Xinyu Du et al. "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation."
       In SIGIR 2023.

   Reference code:
       https://github.com/sudaada/FEARec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.fearec.FEARec
   hopwise.model.sequential_recommender.fearec.HybridAttention
   hopwise.model.sequential_recommender.fearec.FeedForward
   hopwise.model.sequential_recommender.fearec.FEABlock
   hopwise.model.sequential_recommender.fearec.FEAEncoder


Module Contents
---------------

.. py:class:: FEARec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: dataset


   .. py:attribute:: config


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: lmd


   .. py:attribute:: lmd_sem


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: same_item_index


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: item_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: ssl


   .. py:attribute:: tau


   .. py:attribute:: sim


   .. py:attribute:: fredom


   .. py:attribute:: fredom_type


   .. py:attribute:: batch_size


   .. py:attribute:: mask_default


   .. py:attribute:: aug_nce_fct


   .. py:attribute:: sem_aug_nce_fct


   .. py:method:: get_same_item_index(dataset)


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: truncated_normal_(tensor, mean=0, std=0.09)


   .. py:method:: get_attention_mask(item_seq)

      Generate left-to-right uni-directional attention mask for multi-head attention.



   .. py:method:: get_bi_attention_mask(item_seq)

      Generate bidirectional attention mask for multi-head attention.



   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: alignment(x, y)
      :staticmethod:



   .. py:method:: uniformity(x)
      :staticmethod:



   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: mask_correlated_samples(batch_size)


   .. py:method:: info_nce(z_i, z_j, temp, batch_size, sim='dot')

      We do not sample negative examples explicitly.
      Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.



   .. py:method:: decompose(z_i, z_j, origin_z, batch_size)

      We do not sample negative examples explicitly.
      Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.



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



.. py:class:: HybridAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, i, config)

   Bases: :py:obj:`torch.nn.Module`


   Hybrid Attention layer: combine time domain self-attention layer and frequency domain attention layer.

   :param input_tensor: the input of the multi-head Hybrid Attention layer
   :type input_tensor: torch.Tensor
   :param attention_mask: the attention mask for input tensor
   :type attention_mask: torch.Tensor

   :returns: the output of the multi-head Hybrid Attention layer
   :rtype: hidden_states (torch.Tensor)


   .. py:attribute:: factor


   .. py:attribute:: scale
      :value: None



   .. py:attribute:: mask_flag
      :value: True



   .. py:attribute:: output_attention
      :value: False



   .. py:attribute:: dropout


   .. py:attribute:: config


   .. py:attribute:: num_attention_heads


   .. py:attribute:: attention_head_size


   .. py:attribute:: all_head_size


   .. py:attribute:: query_layer


   .. py:attribute:: key_layer


   .. py:attribute:: value_layer


   .. py:attribute:: attn_dropout


   .. py:attribute:: dense


   .. py:attribute:: LayerNorm


   .. py:attribute:: out_dropout


   .. py:attribute:: filter_mixer
      :value: None



   .. py:attribute:: global_ratio


   .. py:attribute:: n_layers


   .. py:attribute:: max_item_list_length


   .. py:attribute:: dual_domain


   .. py:attribute:: slide_step


   .. py:attribute:: local_ratio


   .. py:attribute:: filter_size


   .. py:attribute:: left


   .. py:attribute:: right


   .. py:attribute:: q_index


   .. py:attribute:: k_index


   .. py:attribute:: v_index


   .. py:attribute:: std


   .. py:method:: transpose_for_scores(x)


   .. py:method:: time_delay_agg_training(values, corr)

      SpeedUp version of Autocorrelation (a batch-normalization style design)
      This is for the training phase.



   .. py:method:: time_delay_agg_inference(values, corr)

      SpeedUp version of Autocorrelation (a batch-normalization style design)
      This is for the inference phase.



   .. py:method:: forward(input_tensor, attention_mask)


.. py:class:: FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

   Bases: :py:obj:`torch.nn.Module`


   Point-wise feed-forward layer is implemented by two dense layers.

   :param input_tensor: the input of the point-wise feed-forward layer
   :type input_tensor: torch.Tensor

   :returns: the output of the point-wise feed-forward layer
   :rtype: hidden_states (torch.Tensor)


   .. py:attribute:: dense_1


   .. py:attribute:: intermediate_act_fn


   .. py:attribute:: dense_2


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: get_hidden_act(act)


   .. py:method:: gelu(x)

      Implementation of the gelu activation function.

      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

          0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

      Also see https://arxiv.org/abs/1606.08415



   .. py:method:: swish(x)


   .. py:method:: forward(input_tensor)


.. py:class:: FEABlock(n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, n, config)

   Bases: :py:obj:`torch.nn.Module`


   One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

   :param hidden_states: the input of the multi-head self-attention sublayer
   :type hidden_states: torch.Tensor
   :param attention_mask: the attention mask for the multi-head self-attention sublayer
   :type attention_mask: torch.Tensor

   :returns:

             The output of the point-wise feed-forward sublayer,
                                                is the output of the transformer layer.
   :rtype: feedforward_output (torch.Tensor)


   .. py:attribute:: hybrid_attention


   .. py:attribute:: feed_forward


   .. py:method:: forward(hidden_states, attention_mask)


.. py:class:: FEAEncoder(n_layers=2, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12, config=None)

   Bases: :py:obj:`torch.nn.Module`


   One TransformerEncoder consists of several TransformerLayers.

   - n_layers(num): num of transformer layers in transformer encoder. Default: 2
   - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
   - hidden_size(num): the input and output hidden size. Default: 64
   - inner_size(num): the dimensionality in feed-forward layer. Default: 256
   - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
   - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
   - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                 candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
   - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12



   .. py:attribute:: n_layers
      :value: 2



   .. py:attribute:: layer


   .. py:method:: forward(hidden_states, attention_mask, output_all_encoded_layers=True)

      :param hidden_states: the input of the TransformerEncoder
      :type hidden_states: torch.Tensor
      :param attention_mask: the attention mask for the input hidden_states
      :type attention_mask: torch.Tensor
      :param output_all_encoded_layers: whether output all transformer layers' output
      :type output_all_encoded_layers: Bool

      :returns: if output_all_encoded_layers is True, return a list consists of all transformer
                layers' output, otherwise return a list only consists of the output of last transformer layer.
      :rtype: all_encoder_layers (list)



