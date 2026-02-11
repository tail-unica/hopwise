hopwise.model.sequential_recommender
====================================

.. py:module:: hopwise.model.sequential_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/sequential_recommender/bert4rec/index
   /autoapi/hopwise/model/sequential_recommender/caser/index
   /autoapi/hopwise/model/sequential_recommender/core/index
   /autoapi/hopwise/model/sequential_recommender/dien/index
   /autoapi/hopwise/model/sequential_recommender/din/index
   /autoapi/hopwise/model/sequential_recommender/fdsa/index
   /autoapi/hopwise/model/sequential_recommender/fearec/index
   /autoapi/hopwise/model/sequential_recommender/fossil/index
   /autoapi/hopwise/model/sequential_recommender/fpmc/index
   /autoapi/hopwise/model/sequential_recommender/gcsan/index
   /autoapi/hopwise/model/sequential_recommender/gru4rec/index
   /autoapi/hopwise/model/sequential_recommender/gru4reccpr/index
   /autoapi/hopwise/model/sequential_recommender/gru4recf/index
   /autoapi/hopwise/model/sequential_recommender/gru4reckg/index
   /autoapi/hopwise/model/sequential_recommender/hgn/index
   /autoapi/hopwise/model/sequential_recommender/hrm/index
   /autoapi/hopwise/model/sequential_recommender/ksr/index
   /autoapi/hopwise/model/sequential_recommender/lightsans/index
   /autoapi/hopwise/model/sequential_recommender/narm/index
   /autoapi/hopwise/model/sequential_recommender/nextitnet/index
   /autoapi/hopwise/model/sequential_recommender/npe/index
   /autoapi/hopwise/model/sequential_recommender/repeatnet/index
   /autoapi/hopwise/model/sequential_recommender/s3rec/index
   /autoapi/hopwise/model/sequential_recommender/sasrec/index
   /autoapi/hopwise/model/sequential_recommender/sasreccpr/index
   /autoapi/hopwise/model/sequential_recommender/sasrecf/index
   /autoapi/hopwise/model/sequential_recommender/shan/index
   /autoapi/hopwise/model/sequential_recommender/sine/index
   /autoapi/hopwise/model/sequential_recommender/srgnn/index
   /autoapi/hopwise/model/sequential_recommender/stamp/index
   /autoapi/hopwise/model/sequential_recommender/transrec/index


Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.BERT4Rec
   hopwise.model.sequential_recommender.Caser
   hopwise.model.sequential_recommender.CORE
   hopwise.model.sequential_recommender.DIEN
   hopwise.model.sequential_recommender.DIN
   hopwise.model.sequential_recommender.FDSA
   hopwise.model.sequential_recommender.FEARec
   hopwise.model.sequential_recommender.FOSSIL
   hopwise.model.sequential_recommender.FPMC
   hopwise.model.sequential_recommender.GCSAN
   hopwise.model.sequential_recommender.GRU4Rec
   hopwise.model.sequential_recommender.GRU4RecCPR
   hopwise.model.sequential_recommender.GRU4RecF
   hopwise.model.sequential_recommender.GRU4RecKG
   hopwise.model.sequential_recommender.HGN
   hopwise.model.sequential_recommender.HRM
   hopwise.model.sequential_recommender.KSR
   hopwise.model.sequential_recommender.LightSANs
   hopwise.model.sequential_recommender.NARM
   hopwise.model.sequential_recommender.NextItNet
   hopwise.model.sequential_recommender.NPE
   hopwise.model.sequential_recommender.RepeatNet
   hopwise.model.sequential_recommender.S3Rec
   hopwise.model.sequential_recommender.SASRec
   hopwise.model.sequential_recommender.SASRecCPR
   hopwise.model.sequential_recommender.SASRecF
   hopwise.model.sequential_recommender.SHAN
   hopwise.model.sequential_recommender.SINE
   hopwise.model.sequential_recommender.SRGNN
   hopwise.model.sequential_recommender.STAMP
   hopwise.model.sequential_recommender.TransRec


Package Contents
----------------

.. py:class:: BERT4Rec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: mask_ratio


   .. py:attribute:: MASK_ITEM_SEQ


   .. py:attribute:: POS_ITEMS


   .. py:attribute:: NEG_ITEMS


   .. py:attribute:: MASK_INDEX


   .. py:attribute:: loss_type


   .. py:attribute:: initializer_range


   .. py:attribute:: mask_token


   .. py:attribute:: mask_item_length


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: output_ffn


   .. py:attribute:: output_gelu


   .. py:attribute:: output_ln


   .. py:attribute:: output_bias


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: reconstruct_test_data(item_seq, item_seq_len)

      Add mask token at the last position according to the lengths of item_seq



   .. py:method:: forward(item_seq)


   .. py:method:: multi_hot_embed(masked_index, max_length)

      For memory, we only need calculate loss for masked position.
      Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
      gathering the masked position hidden representation.

      .. rubric:: Examples

      sequence: [1 2 3 4 5]

      masked_sequence: [1 mask 3 mask 5]

      masked_index: [1, 3]

      max_length: 5

      multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]



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



.. py:class:: Caser(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   Caser is a model that incorporate CNN for recommendation.

   .. note::

      We did not use the sliding window to generate training instances as in the paper, in order that
      the generation method we used is common to other sequential models.
      For comparison with other models, we set the parameter T in the paper as 1.
      In addition, to prevent excessive CNN layers (ValueError: Training loss is nan), please make sure the parameters MAX_ITEM_LIST_LENGTH small, such as 10.


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: n_h


   .. py:attribute:: n_v


   .. py:attribute:: dropout_prob


   .. py:attribute:: reg_weight


   .. py:attribute:: n_users


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: conv_v


   .. py:attribute:: conv_h


   .. py:attribute:: fc1_dim_v


   .. py:attribute:: fc1_dim_h


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: dropout


   .. py:attribute:: ac_conv


   .. py:attribute:: ac_fc


   .. py:attribute:: reg_loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item_seq)


   .. py:method:: reg_loss_conv_h()

      L2 loss on conv_h



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



.. py:class:: FDSA(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   FDSA is similar with the GRU4RecF implemented in hopwise, which uses two different Transformer encoders to
   encode items and features respectively and concatenates the two subparts' outputs as the final output.



   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: selected_features


   .. py:attribute:: pooling_mode


   .. py:attribute:: device


   .. py:attribute:: num_feature_field


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: feature_embed_layer


   .. py:attribute:: item_trm_encoder


   .. py:attribute:: feature_att_layer


   .. py:attribute:: feature_trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: concat_layer


   .. py:attribute:: other_parameter_name
      :value: ['feature_embed_layer']



   .. py:method:: _init_weights(module)

      Initialize the weights



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



.. py:class:: FOSSIL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   FOSSIL uses similarity of the items as main purpose and uses high MC as a way of sequential preference improve of
   ability of sequential recommendation



   .. py:attribute:: n_users


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: order_len


   .. py:attribute:: reg_weight


   .. py:attribute:: alpha


   .. py:attribute:: item_embedding


   .. py:attribute:: user_lambda


   .. py:attribute:: lambda_


   .. py:attribute:: loss_type


   .. py:method:: inverse_seq_item_embedding(seq_item_embedding, seq_item_len)

      Inverse seq_item_embedding like this (simple to 2-dim):

      [1,2,3,0,0,0] -- ??? -- >> [0,0,0,1,2,3]

      first: [0,0,0,0,0,0] concat [1,2,3,0,0,0]

      using gather_indexes: to get one by one

      first get 3,then 2,last 1



   .. py:method:: reg_loss(user_embedding, item_embedding, seq_output)


   .. py:method:: init_weights(module)


   .. py:method:: forward(seq_item, seq_item_len, user)


   .. py:method:: get_high_order_Markov(high_order_item_embedding, user)

      In order to get the inference of past items and the user's taste to the current predict item



   .. py:method:: get_similarity(seq_item_embedding, seq_item_len)

      In order to get the inference of past items to the current predict item



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



.. py:class:: FPMC(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   The FPMC model is mainly used in the recommendation system to predict the possibility of
   unknown items arousing user interest, and to discharge the item recommendation list.

   .. note::

      In order that the generation method we used is common to other sequential models,
      We set the size of the basket mentioned in the paper equal to 1.
      For comparison with other models, the loss function used is BPR.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: n_users


   .. py:attribute:: UI_emb


   .. py:attribute:: IU_emb


   .. py:attribute:: LI_emb


   .. py:attribute:: IL_emb


   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item_seq, item_seq_len, next_item)


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



.. py:class:: GCSAN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   GCSAN captures rich local dependencies via graph neural network,
    and learns long-range dependencies by applying the self-attention mechanism.

   .. note::

      In the original paper, the attention mechanism in the self-attention layer is a single head,
      for the reusability of the project code, we use a unified transformer component.
      According to the experimental results, we only applied regularization to embedding.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: step


   .. py:attribute:: device


   .. py:attribute:: weight


   .. py:attribute:: reg_weight


   .. py:attribute:: loss_type


   .. py:attribute:: initializer_range


   .. py:attribute:: item_embedding


   .. py:attribute:: gnn


   .. py:attribute:: self_attention


   .. py:attribute:: reg_loss


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _get_slice(item_seq)


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



.. py:class:: GRU4RecCPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   GRU4Rec is a model that incorporate RNN for recommendation.

   .. note::

      Regarding the innovation of this article,we can only achieve the data augmentation mentioned
      in the paper and directly output the embedding of the item,
      in order that the generation method we used is common to other sequential models.


   .. py:attribute:: hidden_size


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: n_facet_all


   .. py:attribute:: n_facet


   .. py:attribute:: n_facet_window


   .. py:attribute:: n_facet_hidden


   .. py:attribute:: n_facet_MLP


   .. py:attribute:: n_facet_context


   .. py:attribute:: n_facet_reranker


   .. py:attribute:: n_facet_emb


   .. py:attribute:: softmax_nonlinear
      :value: 'None'



   .. py:attribute:: use_out_emb


   .. py:attribute:: only_compute_loss
      :value: True



   .. py:attribute:: dense


   .. py:attribute:: n_embd


   .. py:attribute:: use_proj_bias


   .. py:attribute:: weight_mode


   .. py:attribute:: context_norm


   .. py:attribute:: post_remove_context


   .. py:attribute:: reranker_merging_mode


   .. py:attribute:: partition_merging_mode


   .. py:attribute:: reranker_CAN_NUM


   .. py:attribute:: candidates_from_previous_reranker
      :value: True



   .. py:attribute:: MLP_linear


   .. py:attribute:: project_arr


   .. py:attribute:: project_emb


   .. py:attribute:: c
      :value: 123



   .. py:attribute:: emb_dropout


   .. py:attribute:: gru_layers


   .. py:attribute:: item_embedding


   .. py:method:: _init_weights(module)


   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: get_facet_emb(input_emb, i)


   .. py:method:: calculate_loss_prob(interaction, only_compute_prob=False)


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



.. py:class:: GRU4RecF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   In the original paper, the authors proposed several architectures. We compared 3 different
   architectures:

       (1)  Concatenate item input and feature input and use single RNN,

       (2)  Concatenate outputs from two different RNNs,

       (3)  Weighted sum of outputs from two different RNNs.

   We implemented the optimal parallel version(2), which uses different RNNs to
   encode items and features respectively and concatenates the two subparts'
   outputs as the final output. The different RNN encoders are trained simultaneously.


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: selected_features


   .. py:attribute:: pooling_mode


   .. py:attribute:: device


   .. py:attribute:: num_feature_field


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: feature_embed_layer


   .. py:attribute:: item_gru_layers


   .. py:attribute:: feature_gru_layers


   .. py:attribute:: dense_layer


   .. py:attribute:: dropout


   .. py:attribute:: other_parameter_name
      :value: ['feature_embed_layer']



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



.. py:class:: GRU4RecKG(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   It is an extension of GRU4Rec, which concatenates item and its corresponding
   pre-trained knowledge graph embedding feature as the input.



   .. py:attribute:: entity_embedding_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: num_layers


   .. py:attribute:: dropout


   .. py:attribute:: freeze_kg


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: item_emb_dropout


   .. py:attribute:: entity_emb_dropout


   .. py:attribute:: item_gru_layers


   .. py:attribute:: entity_gru_layers


   .. py:attribute:: dense_layer


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



.. py:class:: HGN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   HGN sets feature gating and instance gating to get the important feature and item for predicting the next item


   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: pool_type


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: w1


   .. py:attribute:: w2


   .. py:attribute:: b


   .. py:attribute:: w3


   .. py:attribute:: w4


   .. py:attribute:: item_embedding_for_prediction


   .. py:attribute:: sigmoid


   .. py:attribute:: loss_type


   .. py:method:: reg_loss(user_embedding, item_embedding, seq_item_embedding)


   .. py:method:: _init_weights(module)


   .. py:method:: feature_gating(seq_item_embedding, user_embedding)

      Choose the features that will be sent to the next stage(more important feature, more focus)



   .. py:method:: instance_gating(user_item, user_embedding)

      Choose the last click items that will influence the prediction( more important more chance to get attention)



   .. py:method:: forward(seq_item, user)


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



.. py:class:: HRM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   HRM can well capture both sequential behavior and users’ general taste by involving transaction and
   user representations in prediction.

   HRM user max- & average- pooling as a good helper.


   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: pooling_type_layer_1


   .. py:attribute:: pooling_type_layer_2


   .. py:attribute:: high_order


   .. py:attribute:: reg_weight


   .. py:attribute:: dropout_prob


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: dropout


   .. py:attribute:: loss_type


   .. py:method:: inverse_seq_item(seq_item, seq_item_len)

      Inverse the seq_item, like this
      [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]



   .. py:method:: _init_weights(module)


   .. py:method:: forward(seq_item, user, seq_item_len)


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



.. py:class:: KSR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   KSR integrates the RNN-based networks with Key-Value Memory Network (KV-MN).
   And it further incorporates knowledge base (KB) information to enhance the semantic representation of KV-MN.



   .. py:attribute:: ENTITY_ID


   .. py:attribute:: RELATION_ID


   .. py:attribute:: n_entities


   .. py:attribute:: n_relations


   .. py:attribute:: entity_embedding_matrix


   .. py:attribute:: relation_embedding_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: loss_type


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: gamma


   .. py:attribute:: device


   .. py:attribute:: freeze_kg


   .. py:attribute:: item_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: gru_layers


   .. py:attribute:: dense


   .. py:attribute:: dense_layer_u


   .. py:attribute:: dense_layer_i


   .. py:attribute:: relation_matrix


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _get_kg_embedding(head)

      Difference:
      We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems.



   .. py:method:: _memory_update_cell(user_memory, update_memory)


   .. py:method:: memory_update(item_seq, item_seq_len)

      Define write operator



   .. py:method:: memory_read(seq_output, user_memory)

      Define read operator



   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: _get_item_comb_embedding(item)


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



.. py:class:: LightSANs(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: k_interests


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: seq_len


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: embedding_layer(item_seq)


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



.. py:class:: NARM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
   and capture the user’s main purpose in the current session.



   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: n_layers


   .. py:attribute:: dropout_probs


   .. py:attribute:: device


   .. py:attribute:: item_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: gru


   .. py:attribute:: a_1


   .. py:attribute:: a_2


   .. py:attribute:: v_t


   .. py:attribute:: ct_dropout


   .. py:attribute:: b


   .. py:attribute:: loss_type


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



.. py:class:: NextItNet(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
   efficiently increase the receptive fields without relying on the pooling operation.
   Also residual block structure is used to ease the optimization for much deeper networks.

   .. note::

      As paper said, for comparison purpose, we only predict the next one item in our evaluation,
      and then stop the generating process. Although the number of parameters in residual block (a) is less
      than it in residual block (b), the performance of b is better than a.
      So in our model, we use residual block (b).
      In addition, when dilations is not equal to 1, the training may be slow. To  speed up the efficiency, please set the parameters "reproducibility" False.


   .. py:attribute:: embedding_size


   .. py:attribute:: residual_channels


   .. py:attribute:: block_num


   .. py:attribute:: dilations


   .. py:attribute:: kernel_size


   .. py:attribute:: reg_weight


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: residual_blocks


   .. py:attribute:: final_layer


   .. py:attribute:: reg_loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(item_seq)


   .. py:method:: reg_loss_rb()

      L2 loss on residual blocks



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



.. py:class:: NPE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   models a user’s click to an item in two terms: the personal preference of the user for the item,
   and the relationships between this item and other items clicked by the user



   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: embedding_seq_item


   .. py:attribute:: relu


   .. py:attribute:: dropout


   .. py:attribute:: loss_type


   .. py:method:: _init_weights(module)


   .. py:method:: forward(seq_item, user)


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



.. py:class:: S3Rec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   S3Rec is the first work to incorporate self-supervised learning in
   sequential recommendation.

   .. note::

      Under this framework, we need reconstruct the pretraining data,
      which would affect the pre-training speed.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: FEATURE_FIELD


   .. py:attribute:: FEATURE_LIST


   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


   .. py:attribute:: mask_ratio


   .. py:attribute:: aap_weight


   .. py:attribute:: mip_weight


   .. py:attribute:: map_weight


   .. py:attribute:: sp_weight


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: n_items


   .. py:attribute:: mask_token


   .. py:attribute:: n_features


   .. py:attribute:: item_feat


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: feature_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: aap_norm


   .. py:attribute:: mip_norm


   .. py:attribute:: map_norm


   .. py:attribute:: sp_norm


   .. py:attribute:: loss_fct


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: _associated_attribute_prediction(sequence_output, feature_embedding)


   .. py:method:: _masked_item_prediction(sequence_output, target_item_emb)


   .. py:method:: _masked_attribute_prediction(sequence_output, feature_embedding)


   .. py:method:: _segment_prediction(context, segment_emb)


   .. py:method:: forward(item_seq, bidirectional=True)


   .. py:method:: pretrain(features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment)

      Pretrain out model using four pre-training tasks:

      1. Associated Attribute Prediction

      2. Masked Item Prediction

      3. Masked Attribute Prediction

      4. Segment Prediction



   .. py:method:: _neg_sample(item_set)


   .. py:method:: _padding_zero_at_left(sequence)


   .. py:method:: reconstruct_pretrain_data(item_seq, item_seq_len)

      Generate pre-training data for the pre-training stage.



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



.. py:class:: SASRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SASRec is the first sequential recommender based on self-attentive mechanism.

   .. note::

      In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
      by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
      using Fully Connected Layer to implement the PFFN.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: _init_weights(module)

      Initialize the weights



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



.. py:class:: SASRecCPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SASRec is the first sequential recommender based on self-attentive mechanism.

   .. note::

      In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
      by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
      using Fully Connected Layer to implement the PFFN.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: n_facet_all


   .. py:attribute:: n_facet


   .. py:attribute:: n_facet_window


   .. py:attribute:: n_facet_hidden


   .. py:attribute:: n_facet_MLP


   .. py:attribute:: n_facet_context


   .. py:attribute:: n_facet_reranker


   .. py:attribute:: n_facet_emb


   .. py:attribute:: weight_mode


   .. py:attribute:: context_norm


   .. py:attribute:: post_remove_context


   .. py:attribute:: partition_merging_mode


   .. py:attribute:: reranker_merging_mode


   .. py:attribute:: reranker_CAN_NUM


   .. py:attribute:: candidates_from_previous_reranker
      :value: True



   .. py:attribute:: softmax_nonlinear
      :value: 'None'



   .. py:attribute:: use_proj_bias


   .. py:attribute:: MLP_linear


   .. py:attribute:: project_arr


   .. py:attribute:: project_emb


   .. py:attribute:: output_probs
      :value: True



   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: get_facet_emb(input_emb, i)


   .. py:method:: _init_weights(module)

      Initialize the weights



   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: calculate_loss_prob(interaction, only_compute_prob=False)


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



.. py:class:: SASRecF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   This is an extension of SASRec, which concatenates item representations and item attribute representations
   as the input to the model.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: selected_features


   .. py:attribute:: pooling_mode


   .. py:attribute:: device


   .. py:attribute:: num_feature_field


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: feature_embed_layer


   .. py:attribute:: trm_encoder


   .. py:attribute:: concat_layer


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:attribute:: other_parameter_name
      :value: ['feature_embed_layer']



   .. py:method:: _init_weights(module)

      Initialize the weights



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



.. py:class:: SHAN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SHAN exploit the Hierarchical Attention Network to get the long-short term preference
   first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose



   .. py:attribute:: n_users


   .. py:attribute:: device


   .. py:attribute:: INVERSE_ITEM_SEQ


   .. py:attribute:: embedding_size


   .. py:attribute:: short_item_length


   .. py:attribute:: reg_weight


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: long_w


   .. py:attribute:: long_b


   .. py:attribute:: long_short_w


   .. py:attribute:: long_short_b


   .. py:attribute:: relu


   .. py:attribute:: loss_type


   .. py:method:: reg_loss(user_embedding, item_embedding)


   .. py:method:: init_weights(module)


   .. py:method:: forward(seq_item, user)


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



   .. py:method:: long_and_short_term_attention_based_pooling_layer(long_short_item_embedding, user_embedding, mask=None)

      Fusing the long term purpose with the short-term preference



   .. py:method:: long_term_attention_based_pooling_layer(seq_item_embedding, user_embedding, mask=None)

      Get the long term purpose of user



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



.. py:class:: SRGNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SRGNN regards the conversation history as a directed graph.
   In addition to considering the connection between the item and the adjacent item,
   it also considers the connection with other interactive items.

   Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

   Outgoing edges:
       === ===== ===== ===== =====
        \    1     2     3     4
       === ===== ===== ===== =====
        1    0     1     0     0
        2    0     0    1/2   1/2
        3    0     1     0     0
        4    0     0     0     0
       === ===== ===== ===== =====

   Incoming edges:
       === ===== ===== ===== =====
        \    1     2     3     4
       === ===== ===== ===== =====
        1    0     0     0     0
        2   1/2    0    1/2    0
        3    0     1     0     0
        4    0     1     0     0
       === ===== ===== ===== =====


   .. py:attribute:: embedding_size


   .. py:attribute:: step


   .. py:attribute:: device


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: gnn


   .. py:attribute:: linear_one


   .. py:attribute:: linear_two


   .. py:attribute:: linear_three


   .. py:attribute:: linear_transform


   .. py:method:: _reset_parameters()


   .. py:method:: _get_slice(item_seq)


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



.. py:class:: TransRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   TransRec is translation-based model for sequential recommendation.
   It assumes that the `prev. item` + `user`  = `next item`.
   We use the Euclidean Distance to calculate the similarity in this implementation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_users


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: bias


   .. py:attribute:: T


   .. py:attribute:: bpr_loss


   .. py:attribute:: emb_loss


   .. py:attribute:: reg_loss


   .. py:method:: _l2_distance(x, y)


   .. py:method:: gather_last_items(item_seq, gather_index)

      Gathers the last_item at the specific positions over a minibatch



   .. py:method:: forward(user, item_seq, item_seq_len)


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



