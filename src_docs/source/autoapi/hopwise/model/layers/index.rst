hopwise.model.layers
====================

.. py:module:: hopwise.model.layers

.. autoapi-nested-parse::

   hopwise.model.layers
   #############################
   Common Layers in recommender system



Classes
-------

.. autoapisummary::

   hopwise.model.layers.MLPLayers
   hopwise.model.layers.FMEmbedding
   hopwise.model.layers.FLEmbedding
   hopwise.model.layers.BaseFactorizationMachine
   hopwise.model.layers.BiGNNLayer
   hopwise.model.layers.AttLayer
   hopwise.model.layers.Dice
   hopwise.model.layers.SequenceAttLayer
   hopwise.model.layers.VanillaAttention
   hopwise.model.layers.MultiHeadAttention
   hopwise.model.layers.FeedForward
   hopwise.model.layers.TransformerLayer
   hopwise.model.layers.TransformerEncoder
   hopwise.model.layers.ItemToInterestAggregation
   hopwise.model.layers.LightMultiHeadAttention
   hopwise.model.layers.LightTransformerLayer
   hopwise.model.layers.LightTransformerEncoder
   hopwise.model.layers.ContextSeqEmbAbstractLayer
   hopwise.model.layers.ContextSeqEmbLayer
   hopwise.model.layers.FeatureSeqEmbLayer
   hopwise.model.layers.CNNLayers
   hopwise.model.layers.FMFirstOrderLinear
   hopwise.model.layers.SparseDropout


Functions
---------

.. autoapisummary::

   hopwise.model.layers.activation_layer


Module Contents
---------------

.. py:class:: MLPLayers(layers, dropout=0.0, activation='relu', bn=False, init_method=None, last_activation=True)

   Bases: :py:obj:`torch.nn.Module`


   MLPLayers

   :param - layers: a list contains the size of each layer in mlp layers
   :type - layers: list
   :param - dropout: probability of an element to be zeroed. Default: 0
   :type - dropout: float
   :param - activation: activation function after each layer in mlp layers. Default: 'relu'.
                        candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'
   :type - activation: str

   Shape:

       - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
         :math:`H_{in}` must equal to the first value in `layers`
       - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

   Examples::

       >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
       >>> input = torch.randn(128, 64)
       >>> output = m(input)
       >>> print(output.size())
       >>> torch.Size([128, 16])


   .. py:attribute:: layers


   .. py:attribute:: dropout
      :value: 0.0



   .. py:attribute:: activation
      :value: 'relu'



   .. py:attribute:: use_bn
      :value: False



   .. py:attribute:: init_method
      :value: None



   .. py:attribute:: mlp_layers


   .. py:method:: init_weights(module)


   .. py:method:: forward(input_feature)


.. py:function:: activation_layer(activation_name='relu', emb_dim=None)

   Construct activation layers

   :param activation_name: str, name of activation function
   :param emb_dim: int, used for Dice activation

   :returns: activation layer
   :rtype: activation


.. py:class:: FMEmbedding(field_dims, offsets, embed_dim)

   Bases: :py:obj:`torch.nn.Module`


   Embedding for token fields.

   :param field_dims: list, the number of tokens in each token fields
   :param offsets: list, the dimension offset of each token field
   :param embed_dim: int, the dimension of output embedding vectors

   Input:
       input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

   :returns: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
   :rtype: output


   .. py:attribute:: embedding


   .. py:attribute:: offsets


   .. py:method:: forward(input_x)


.. py:class:: FLEmbedding(field_dims, offsets, embed_dim)

   Bases: :py:obj:`torch.nn.Module`


   Embedding for float fields.

   :param field_dims: list, the number of float in each float fields
   :param offsets: list, the dimension offset of each float field
   :param embed_dim: int, the dimension of output embedding vectors

   Input:
       input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,2)``.

   :returns: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
   :rtype: output


   .. py:attribute:: embedding


   .. py:attribute:: offsets


   .. py:method:: forward(input_x)


.. py:class:: BaseFactorizationMachine(reduce_sum=True)

   Bases: :py:obj:`torch.nn.Module`


   Calculate FM result over the embeddings

   :param reduce_sum: bool, whether to sum the result, default is True.

   Input:
       input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

   Output
       output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.


   .. py:attribute:: reduce_sum
      :value: True



   .. py:method:: forward(input_x)


.. py:class:: BiGNNLayer(in_dim, out_dim)

   Bases: :py:obj:`torch.nn.Module`


   Propagate a layer of Bi-interaction GNN

   .. math::
       output = (L+I)EW_1 + LE \otimes EW_2


   .. py:attribute:: in_dim


   .. py:attribute:: out_dim


   .. py:attribute:: linear


   .. py:attribute:: interActTransform


   .. py:method:: forward(lap_matrix, eye_matrix, features)


.. py:class:: AttLayer(in_dim, att_dim)

   Bases: :py:obj:`torch.nn.Module`


   Calculate the attention signal(weight) according the input tensor.

   :param infeatures: A 3D input tensor with shape of[batch_size, M, embed_dim].
   :type infeatures: torch.FloatTensor

   :returns: Attention weight of input. shape of [batch_size, M].
   :rtype: torch.FloatTensor


   .. py:attribute:: in_dim


   .. py:attribute:: att_dim


   .. py:attribute:: w


   .. py:attribute:: h


   .. py:method:: forward(infeatures)


.. py:class:: Dice(emb_size)

   Bases: :py:obj:`torch.nn.Module`


   Dice activation function

   .. math::
       f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

   .. math::
       p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}


   .. py:attribute:: sigmoid


   .. py:attribute:: alpha


   .. py:method:: forward(score)


.. py:class:: SequenceAttLayer(mask_mat, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=False, return_seq_weight=True)

   Bases: :py:obj:`torch.nn.Module`


   Attention Layer. Get the representation of each user in the batch.

   :param queries: candidate ads, [B, H], H means embedding_size * feat_num
   :type queries: torch.Tensor
   :param keys: user_hist, [B, T, H]
   :type keys: torch.Tensor
   :param keys_length: mask, [B]
   :type keys_length: torch.Tensor

   :returns: result
   :rtype: torch.Tensor


   .. py:attribute:: att_hidden_size
      :value: (80, 40)



   .. py:attribute:: activation
      :value: 'sigmoid'



   .. py:attribute:: softmax_stag
      :value: False



   .. py:attribute:: return_seq_weight
      :value: True



   .. py:attribute:: mask_mat


   .. py:attribute:: att_mlp_layers


   .. py:attribute:: dense


   .. py:method:: forward(queries, keys, keys_length)


.. py:class:: VanillaAttention(hidden_dim, attn_dim)

   Bases: :py:obj:`torch.nn.Module`


   Vanilla attention layer is implemented by linear layer.

   :param input_tensor: the input of the attention layer
   :type input_tensor: torch.Tensor

   :returns: the outputs of the attention layer
             weights (torch.Tensor): the attention weights
   :rtype: hidden_states (torch.Tensor)


   .. py:attribute:: projection


   .. py:method:: forward(input_tensor)


.. py:class:: MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)

   Bases: :py:obj:`torch.nn.Module`


   Multi-head Self-attention layers, a attention score dropout layer is introduced.

   :param input_tensor: the input of the multi-head self-attention layer
   :type input_tensor: torch.Tensor
   :param attention_mask: the attention mask for input tensor
   :type attention_mask: torch.Tensor

   :returns: the output of the multi-head self-attention layer
   :rtype: hidden_states (torch.Tensor)


   .. py:attribute:: num_attention_heads


   .. py:attribute:: attention_head_size


   .. py:attribute:: all_head_size


   .. py:attribute:: sqrt_attention_head_size


   .. py:attribute:: query


   .. py:attribute:: key


   .. py:attribute:: value


   .. py:attribute:: softmax


   .. py:attribute:: attn_dropout


   .. py:attribute:: dense


   .. py:attribute:: LayerNorm


   .. py:attribute:: out_dropout


   .. py:method:: transpose_for_scores(x)


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


.. py:class:: TransformerLayer(n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)

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


   .. py:attribute:: multi_head_attention


   .. py:attribute:: feed_forward


   .. py:method:: forward(hidden_states, attention_mask)


.. py:class:: TransformerEncoder(n_layers=2, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12)

   Bases: :py:obj:`torch.nn.Module`


   One TransformerEncoder consists of several TransformerLayers.

   :param n_layers: num of transformer layers in transformer encoder. Default: 2
   :type n_layers: num
   :param n_heads: num of attention heads for multi-head attention layer. Default: 2
   :type n_heads: num
   :param hidden_size: the input and output hidden size. Default: 64
   :type hidden_size: num
   :param inner_size: the dimensionality in feed-forward layer. Default: 256
   :type inner_size: num
   :param hidden_dropout_prob: probability of an element to be zeroed. Default: 0.5
   :type hidden_dropout_prob: float
   :param attn_dropout_prob: probability of an attention score to be zeroed. Default: 0.5
   :type attn_dropout_prob: float
   :param hidden_act: activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
   :type hidden_act: str
   :param layer_norm_eps: a value added to the denominator for numerical stability. Default: 1e-12
   :type layer_norm_eps: float


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



.. py:class:: ItemToInterestAggregation(seq_len, hidden_size, k_interests=5)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: k_interests
      :value: 5



   .. py:attribute:: theta


   .. py:method:: forward(input_tensor)


.. py:class:: LightMultiHeadAttention(n_heads, k_interests, hidden_size, seq_len, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: num_attention_heads


   .. py:attribute:: attention_head_size


   .. py:attribute:: all_head_size


   .. py:attribute:: query


   .. py:attribute:: key


   .. py:attribute:: value


   .. py:attribute:: attpooling_key


   .. py:attribute:: attpooling_value


   .. py:attribute:: attn_scale_factor
      :value: 2



   .. py:attribute:: pos_q_linear


   .. py:attribute:: pos_k_linear


   .. py:attribute:: pos_scaling


   .. py:attribute:: pos_ln


   .. py:attribute:: attn_dropout


   .. py:attribute:: dense


   .. py:attribute:: LayerNorm


   .. py:attribute:: out_dropout


   .. py:method:: transpose_for_scores(x)


   .. py:method:: forward(input_tensor, pos_emb)


.. py:class:: LightTransformerLayer(n_heads, k_interests, hidden_size, seq_len, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)

   Bases: :py:obj:`torch.nn.Module`


   One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

   :param hidden_states: the input of the multi-head self-attention sublayer
   :type hidden_states: torch.Tensor
   :param attention_mask: the attention mask for the multi-head self-attention sublayer
   :type attention_mask: torch.Tensor

   :returns: the output of the point-wise feed-forward sublayer, is the output of the transformer layer
   :rtype: feedforward_output (torch.Tensor)


   .. py:attribute:: multi_head_attention


   .. py:attribute:: feed_forward


   .. py:method:: forward(hidden_states, pos_emb)


.. py:class:: LightTransformerEncoder(n_layers=2, n_heads=2, k_interests=5, hidden_size=64, seq_len=50, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12)

   Bases: :py:obj:`torch.nn.Module`


   One LightTransformerEncoder consists of several LightTransformerLayers.

   :param n_layers: num of transformer layers in transformer encoder. Default: 2
   :type n_layers: num
   :param n_heads: num of attention heads for multi-head attention layer. Default: 2
   :type n_heads: num
   :param hidden_size: the input and output hidden size. Default: 64
   :type hidden_size: num
   :param inner_size: the dimensionality in feed-forward layer. Default: 256
   :type inner_size: num
   :param hidden_dropout_prob: probability of an element to be zeroed. Default: 0.5
   :type hidden_dropout_prob: float
   :param attn_dropout_prob: probability of an attention score to be zeroed. Default: 0.5
   :type attn_dropout_prob: float
   :param hidden_act: activation function in feed-forward layer. Default: 'gelu'.
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
   :type hidden_act: str
   :param layer_norm_eps: a value added to the denominator for numerical stability. Default: 1e-12
   :type layer_norm_eps: float


   .. py:attribute:: layer


   .. py:method:: forward(hidden_states, pos_emb, output_all_encoded_layers=True)

      :param hidden_states: the input of the TrandformerEncoder
      :type hidden_states: torch.Tensor
      :param attention_mask: the attention mask for the input hidden_states
      :type attention_mask: torch.Tensor
      :param output_all_encoded_layers: whether output all transformer layers' output
      :type output_all_encoded_layers: Bool

      :returns: if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
                otherwise return a list only consists of the output of last transformer layer.
      :rtype: all_encoder_layers (list)



.. py:class:: ContextSeqEmbAbstractLayer

   Bases: :py:obj:`torch.nn.Module`


   For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices.


   .. py:attribute:: token_field_offsets


   .. py:attribute:: float_field_offsets


   .. py:attribute:: token_embedding_table


   .. py:attribute:: float_embedding_table


   .. py:attribute:: token_seq_embedding_table


   .. py:attribute:: float_seq_embedding_table


   .. py:attribute:: token_field_names
      :value: None



   .. py:attribute:: token_field_dims
      :value: None



   .. py:attribute:: float_field_names
      :value: None



   .. py:attribute:: float_field_dims
      :value: None



   .. py:attribute:: token_seq_field_names
      :value: None



   .. py:attribute:: token_seq_field_dims
      :value: None



   .. py:attribute:: float_seq_field_names
      :value: None



   .. py:attribute:: float_seq_field_dims
      :value: None



   .. py:attribute:: num_feature_field
      :value: None



   .. py:method:: get_fields_name_dim()

      Get user feature field and item feature field.



   .. py:method:: get_embedding()

      Get embedding of all features.



   .. py:method:: embed_float_fields(float_fields, type, embed=True)

      Get the embedding of float fields.
      In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
      when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

      :param float_fields: [batch_size, max_item_length, num_float_field]
      :type float_fields: torch.Tensor
      :param type: user or item
      :type type: str
      :param embed: embed or not
      :type embed: bool

      :returns: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]
      :rtype: torch.Tensor



   .. py:method:: embed_token_fields(token_fields, type)

      Get the embedding of token fields

      :param token_fields: input, [batch_size, max_item_length, num_token_field]
      :type token_fields: torch.Tensor
      :param type: user or item
      :type type: str

      :returns: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]
      :rtype: torch.Tensor



   .. py:method:: embed_float_seq_fields(float_seq_fields, type)

      Embed the float sequence feature columns

      :param float_seq_fields: The input tensor. shape of [batch_size, seq_len, 2]
      :type float_seq_fields: torch.FloatTensor
      :param mode: How to aggregate the embedding of feature in this field. default=mean
      :type mode: str

      :returns: The result embedding tensor of float sequence columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_token_seq_fields(token_seq_fields, type)

      Get the embedding of token_seq fields.

      :param token_seq_fields: input, [batch_size, max_item_length, seq_len]`
      :type token_seq_fields: torch.Tensor
      :param type: user or item
      :type type: str
      :param mode: mean/max/sum
      :type mode: str

      :returns: result [batch_size, max_item_length, num_token_seq_field, embed_dim]
      :rtype: torch.Tensor



   .. py:method:: embed_input_fields(user_idx, item_idx)

      Get the embedding of user_idx and item_idx

      :param user_idx: interaction['user_id']
      :type user_idx: torch.Tensor
      :param item_idx: interaction['item_id_list']
      :type item_idx: torch.Tensor

      :returns: embedding of user feature and item feature
      :rtype: dict



   .. py:method:: forward(user_idx, item_idx)


.. py:class:: ContextSeqEmbLayer(dataset, embedding_size, pooling_mode, device)

   Bases: :py:obj:`ContextSeqEmbAbstractLayer`


   For Deep Interest Network, return all features (including user features and item features) embedding matrices.


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: dataset


   .. py:attribute:: user_feat


   .. py:attribute:: item_feat


   .. py:attribute:: field_names


   .. py:attribute:: types
      :value: ['user', 'item']



   .. py:attribute:: pooling_mode


.. py:class:: FeatureSeqEmbLayer(dataset, embedding_size, selected_features, pooling_mode, device)

   Bases: :py:obj:`ContextSeqEmbAbstractLayer`


   For feature-rich sequential recommenders, return item features embedding matrices according to
   selected features.


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: dataset


   .. py:attribute:: user_feat
      :value: None



   .. py:attribute:: item_feat


   .. py:attribute:: field_names


   .. py:attribute:: types
      :value: ['item']



   .. py:attribute:: pooling_mode


.. py:class:: CNNLayers(channels, kernels, strides, activation='relu', init_method=None)

   Bases: :py:obj:`torch.nn.Module`


   CNNLayers

   :param - channels: a list contains the channels of each layer in cnn layers
   :type - channels: list
   :param - kernel: a list contains the kernels of each layer in cnn layers
   :type - kernel: list
   :param - strides: a list contains the channels of each layer in cnn layers
   :type - strides: list
   :param - activation: activation function after each layer in mlp layers. Default: 'relu'
                        candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'
   :type - activation: str

   Shape:
       - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
       - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

       .. math::
           H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                     \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

       .. math::
           W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                     \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

   Examples::

       >>> m = CNNLayers([1, 32, 32], [2,2], [2,2], 'relu')
       >>> input = torch.randn(128, 1, 64, 64)
       >>> output = m(input)
       >>> print(output.size())
       >>> torch.Size([128, 32, 16, 16])


   .. py:attribute:: channels


   .. py:attribute:: kernels


   .. py:attribute:: strides


   .. py:attribute:: activation
      :value: 'relu'



   .. py:attribute:: init_method
      :value: None



   .. py:attribute:: num_of_nets


   .. py:attribute:: cnn_layers


   .. py:method:: init_weights(module)


   .. py:method:: forward(input_feature)


.. py:class:: FMFirstOrderLinear(config, dataset, output_dim=1)

   Bases: :py:obj:`torch.nn.Module`


   Calculate the first order score of the input features.
   This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.



   .. py:attribute:: field_names


   .. py:attribute:: LABEL


   .. py:attribute:: device


   .. py:attribute:: numerical_features


   .. py:attribute:: token_field_names
      :value: []



   .. py:attribute:: token_field_dims
      :value: []



   .. py:attribute:: float_field_names
      :value: []



   .. py:attribute:: float_field_dims
      :value: []



   .. py:attribute:: token_seq_field_names
      :value: []



   .. py:attribute:: token_seq_field_dims
      :value: []



   .. py:attribute:: float_seq_field_names
      :value: []



   .. py:attribute:: float_seq_field_dims
      :value: []



   .. py:attribute:: bias


   .. py:method:: embed_float_fields(float_fields)

      Embed the float feature columns

      :param float_fields: The input dense tensor. shape of [batch_size, num_float_field, 2]
      :type float_fields: torch.FloatTensor
      :param embed: Return the embedding of columns or just the columns itself. Defaults to ``True``.
      :type embed: bool

      :returns: The result embedding tensor of float columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_float_seq_fields(float_seq_fields, mode='mean')

      Embed the float sequence feature columns

      :param float_seq_fields: The input tensor. shape of [batch_size, seq_len, 2]
      :type float_seq_fields: torch.LongTensor
      :param mode: How to aggregate the embedding of feature in this field. default=mean
      :type mode: str

      :returns: The result embedding tensor of float sequence columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_token_fields(token_fields)

      Calculate the first order score of token feature columns

      :param token_fields: The input tensor. shape of [batch_size, num_token_field]
      :type token_fields: torch.LongTensor

      :returns: The first order score of token feature columns
      :rtype: torch.FloatTensor



   .. py:method:: embed_token_seq_fields(token_seq_fields)

      Calculate the first order score of token sequence feature columns

      :param token_seq_fields: The input tensor. shape of [batch_size, seq_len]
      :type token_seq_fields: torch.LongTensor

      :returns: The first order score of token sequence feature columns
      :rtype: torch.FloatTensor



   .. py:method:: forward(interaction)


.. py:class:: SparseDropout(p=0.5)

   Bases: :py:obj:`torch.nn.Module`


   This is a Module that execute Dropout on Pytorch sparse tensor.


   .. py:attribute:: kprob
      :value: 0.5



   .. py:method:: forward(x)


