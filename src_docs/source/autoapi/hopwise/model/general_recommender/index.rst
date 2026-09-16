hopwise.model.general_recommender
=================================

.. py:module:: hopwise.model.general_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/general_recommender/admmslim/index
   /autoapi/hopwise/model/general_recommender/bpr/index
   /autoapi/hopwise/model/general_recommender/cdae/index
   /autoapi/hopwise/model/general_recommender/convncf/index
   /autoapi/hopwise/model/general_recommender/dgcf/index
   /autoapi/hopwise/model/general_recommender/diffrec/index
   /autoapi/hopwise/model/general_recommender/dmf/index
   /autoapi/hopwise/model/general_recommender/ease/index
   /autoapi/hopwise/model/general_recommender/enmf/index
   /autoapi/hopwise/model/general_recommender/fism/index
   /autoapi/hopwise/model/general_recommender/gcmc/index
   /autoapi/hopwise/model/general_recommender/itemknn/index
   /autoapi/hopwise/model/general_recommender/ldiffrec/index
   /autoapi/hopwise/model/general_recommender/lightgcn/index
   /autoapi/hopwise/model/general_recommender/line/index
   /autoapi/hopwise/model/general_recommender/macridvae/index
   /autoapi/hopwise/model/general_recommender/multidae/index
   /autoapi/hopwise/model/general_recommender/multivae/index
   /autoapi/hopwise/model/general_recommender/nais/index
   /autoapi/hopwise/model/general_recommender/nceplrec/index
   /autoapi/hopwise/model/general_recommender/ncl/index
   /autoapi/hopwise/model/general_recommender/neumf/index
   /autoapi/hopwise/model/general_recommender/ngcf/index
   /autoapi/hopwise/model/general_recommender/nncf/index
   /autoapi/hopwise/model/general_recommender/pop/index
   /autoapi/hopwise/model/general_recommender/ract/index
   /autoapi/hopwise/model/general_recommender/random/index
   /autoapi/hopwise/model/general_recommender/recvae/index
   /autoapi/hopwise/model/general_recommender/sgl/index
   /autoapi/hopwise/model/general_recommender/simplex/index
   /autoapi/hopwise/model/general_recommender/slimelastic/index
   /autoapi/hopwise/model/general_recommender/spectralcf/index


Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.ADMMSLIM
   hopwise.model.general_recommender.BPR
   hopwise.model.general_recommender.CDAE
   hopwise.model.general_recommender.ConvNCF
   hopwise.model.general_recommender.DGCF
   hopwise.model.general_recommender.DiffRec
   hopwise.model.general_recommender.DMF
   hopwise.model.general_recommender.EASE
   hopwise.model.general_recommender.ENMF
   hopwise.model.general_recommender.FISM
   hopwise.model.general_recommender.GCMC
   hopwise.model.general_recommender.ItemKNN
   hopwise.model.general_recommender.LDiffRec
   hopwise.model.general_recommender.LightGCN
   hopwise.model.general_recommender.LINE
   hopwise.model.general_recommender.MacridVAE
   hopwise.model.general_recommender.MultiDAE
   hopwise.model.general_recommender.MultiVAE
   hopwise.model.general_recommender.NAIS
   hopwise.model.general_recommender.NCEPLRec
   hopwise.model.general_recommender.NCL
   hopwise.model.general_recommender.NeuMF
   hopwise.model.general_recommender.NGCF
   hopwise.model.general_recommender.NNCF
   hopwise.model.general_recommender.Pop
   hopwise.model.general_recommender.RaCT
   hopwise.model.general_recommender.Random
   hopwise.model.general_recommender.RecVAE
   hopwise.model.general_recommender.SGL
   hopwise.model.general_recommender.SimpleX
   hopwise.model.general_recommender.SLIMElastic
   hopwise.model.general_recommender.SpectralCF


Package Contents
----------------

.. py:class:: ADMMSLIM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   This is a abstract general recommender. All the general model should implement this class.
   The base general recommender class provide the basic dataset and parameters information.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: dummy_param


   .. py:attribute:: center_columns


   .. py:attribute:: item_means


   .. py:attribute:: item_similarity


   .. py:attribute:: interaction_matrix


   .. py:method:: forward()


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



.. py:class:: BPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   BPR is a basic matrix factorization model that be trained in the pairwise way.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: loss


   .. py:method:: get_user_embedding(user)

      Get a batch of user embedding tensor according to input user's id.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor

      :returns: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
      :rtype: torch.FloatTensor



   .. py:method:: get_item_embedding(item)

      Get a batch of item embedding tensor according to input item's id.

      :param item: The input tensor that contains item's id, shape: [batch_size, ]
      :type item: torch.LongTensor

      :returns: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
      :rtype: torch.FloatTensor



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



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



.. py:class:: CDAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   Collaborative Denoising Auto-Encoder (CDAE) is a recommendation model
   for top-N recommendation that utilizes the idea of Denoising Auto-Encoders.
   We implement the the CDAE model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: reg_weight_1


   .. py:attribute:: reg_weight_2


   .. py:attribute:: loss_type


   .. py:attribute:: hid_activation


   .. py:attribute:: out_activation


   .. py:attribute:: embedding_size


   .. py:attribute:: corruption_ratio


   .. py:attribute:: dropout


   .. py:attribute:: h_user


   .. py:attribute:: h_item


   .. py:attribute:: out_layer


   .. py:method:: forward(x_items, x_users)


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



.. py:class:: ConvNCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   ConvNCF is a a new neural network framework for collaborative filtering based on NCF.
   It uses an outer product operation above the embedding layer,
   which results in a semantic-rich interaction map that encodes pairwise correlations between embedding dimensions.
   We carefully design the data interface and use sparse tensor to train and test efficiently.
   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: cnn_channels


   .. py:attribute:: cnn_kernels


   .. py:attribute:: cnn_strides


   .. py:attribute:: dropout_prob


   .. py:attribute:: regs


   .. py:attribute:: train_method


   .. py:attribute:: pre_model_path


   .. py:attribute:: cnn_layers


   .. py:attribute:: predict_layers


   .. py:attribute:: loss


   .. py:method:: forward(user, item)


   .. py:method:: reg_loss()

      Calculate the L2 normalization loss of model parameters.
      Including embedding matrices and weight matrices of model.

      :returns: The L2 Loss tensor. shape of [1,]
      :rtype: loss(torch.FloatTensor)



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



.. py:class:: DGCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   DGCF is a disentangled representation enhanced matrix factorization model.
   The interaction matrix of :math:`n_{users} \times n_{items}` is decomposed to :math:`n_{factors}` intent graph,
   we carefully design the data interface and use sparse tensor to train and test efficiently.
   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: interaction_matrix


   .. py:attribute:: embedding_size


   .. py:attribute:: n_factors


   .. py:attribute:: n_iterations


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: cor_weight


   .. py:attribute:: cor_batch_size


   .. py:attribute:: all_h_list


   .. py:attribute:: all_t_list


   .. py:attribute:: edge2head


   .. py:attribute:: head2edge


   .. py:attribute:: tail2edge


   .. py:attribute:: edge2head_mat


   .. py:attribute:: head2edge_mat


   .. py:attribute:: tail2edge_mat


   .. py:attribute:: num_edge


   .. py:attribute:: num_node


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: softmax


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: _build_sparse_tensor(indices, values, size)


   .. py:method:: _get_ego_embeddings()


   .. py:method:: build_matrix(A_values)

      Get the normalized interaction matrix of users and items according to A_values.

      Construct the square matrix from the training data and normalize it
      using the laplace matrix.

      :param A_values: (num_edge, n_factors)
      :type A_values: torch.cuda.FloatTensor

      .. math::
          A_{hat} = D^{-0.5} \times A \times D^{-0.5}

      :returns: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
      :rtype: torch.cuda.FloatTensor



   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: create_cor_loss(cor_u_embeddings, cor_i_embeddings)

      Calculate the correlation loss for a sampled users and items.

      :param cor_u_embeddings: (cor_batch_size, n_factors)
      :type cor_u_embeddings: torch.cuda.FloatTensor
      :param cor_i_embeddings: (cor_batch_size, n_factors)
      :type cor_i_embeddings: torch.cuda.FloatTensor

      :returns: correlation loss.
      :rtype: torch.Tensor



   .. py:method:: _create_distance_correlation(X1, X2)


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



.. py:class:: DiffRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   DiffRec is a generative recommender model which infers users' interaction probabilities in a denoising manner.
   Note that DiffRec simultaneously ranks all items for each user.
   We implement the the DiffRec model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: time_aware


   .. py:attribute:: w_max


   .. py:attribute:: w_min


   .. py:attribute:: noise_schedule


   .. py:attribute:: noise_scale


   .. py:attribute:: noise_min


   .. py:attribute:: noise_max


   .. py:attribute:: steps


   .. py:attribute:: beta_fixed


   .. py:attribute:: emb_size


   .. py:attribute:: norm


   .. py:attribute:: reweight


   .. py:attribute:: sampling_noise


   .. py:attribute:: sampling_steps


   .. py:attribute:: mlp_act_func


   .. py:attribute:: history_num_per_term


   .. py:attribute:: Lt_history


   .. py:attribute:: Lt_count


   .. py:attribute:: mlp


   .. py:method:: build_histroy_items(dataset)

      Add time-aware reweighting to the original user-item interaction matrix when config['time-aware'] is True.



   .. py:method:: get_betas()

      Given the schedule name, create the betas for the diffusion process.



   .. py:method:: calculate_for_diffusion()

      Calculate the coefficients for the diffusion process.



   .. py:method:: p_sample(x_start)

      Generate users' interaction probabilities in a denoising manner.

      :param x_start: the input tensor that contains user's history interaction matrix,
                      for DiffRec shape: [batch_size, n_items]
                      for LDiffRec shape: [batch_size, hidden_size]
      :type x_start: torch.FloatTensor

      :returns:

                the interaction probabilities,
                                   for DiffRec shape: [batch_size, n_items]
                                   for LDiffRec shape: [batch_size, hidden_size]
      :rtype: torch.FloatTensor



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



   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: reweight_loss(x_start, x_t, mse, ts, target, model_output, device)


   .. py:method:: update_Lt_history(ts, reloss)


   .. py:method:: sample_timesteps(batch_size, device, method='uniform', uniform_prob=0.001)


   .. py:method:: q_sample(x_start, t, noise=None)


   .. py:method:: q_posterior_mean_variance(x_start, x_t, t)

      Compute the mean and variance of the diffusion posterior:
      q(x_{t-1} | x_t, x_0)



   .. py:method:: p_mean_variance(x, t)

      Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
      the initial x, x_0.



   .. py:method:: _predict_xstart_from_eps(x_t, t, eps)


   .. py:method:: SNR(t)

      Compute the signal-to-noise ratio for a single timestep.



   .. py:method:: _extract_into_tensor(arr, timesteps, broadcast_shape)

      Extract values from a 1-D torch tensor for a batch of indices.

      :param arr: the 1-D torch tensor.
      :type arr: torch.Tensor
      :param timesteps: a tensor of indices into the array to extract.
      :type timesteps: torch.Tensor
      :param broadcast_shape: a larger shape of K dimensions with the batch
                              dimension equal to the length of timesteps.
      :type broadcast_shape: torch.Size

      :returns: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
      :rtype: torch.Tensor



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



.. py:class:: EASE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   EASE is a linear model for collaborative filtering, which combines the
   strengths of auto-encoders and neighborhood-based approaches.



   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: dummy_param


   .. py:attribute:: item_similarity


   .. py:attribute:: interaction_matrix


   .. py:attribute:: other_parameter_name
      :value: ['interaction_matrix', 'item_similarity']



   .. py:attribute:: device


   .. py:method:: forward()


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



.. py:class:: ENMF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   ENMF is an efficient non-sampling model for general recommendation.
   In order to run non-sampling model, please set the neg_sampling parameter as None .



   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: reg_weight


   .. py:attribute:: negative_weight


   .. py:attribute:: history_item_matrix


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: H_i


   .. py:attribute:: dropout


   .. py:method:: reg_loss()

      Calculate the reg loss for embedding layers and mlp layers

      :returns: reg loss
      :rtype: torch.Tensor



   .. py:method:: forward(user)


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



.. py:class:: FISM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   FISM is an item-based model for generating top-N recommendations that learns the
   item-item similarity matrix as the product of two low dimensional latent factor matrices.
   These matrices are learned using a structural equation modeling approach, where in the
   value being estimated is not used for its own estimation.



   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weights


   .. py:attribute:: alpha


   .. py:attribute:: split_to


   .. py:attribute:: item_src_embedding


   .. py:attribute:: item_dst_embedding


   .. py:attribute:: user_bias


   .. py:attribute:: item_bias


   .. py:attribute:: bceloss


   .. py:method:: get_history_info(dataset)

      Get the user history interaction information

      :param dataset: train dataset
      :type dataset: DataSet

      :returns: (history_item_matrix, history_lens, mask_mat)
      :rtype: tuple



   .. py:method:: reg_loss()

      Calculate the reg loss for embedding layers

      :returns: reg loss
      :rtype: torch.Tensor



   .. py:method:: _init_weights(module)

      Initialize the module's parameters

      .. note::

         It's a little different from the source code, because pytorch has no function to initialize
         the parameters by truncated normal distribution, so we replace it with xavier normal distribution



   .. py:method:: inter_forward(user, item)

      Forward the model by interaction



   .. py:method:: user_forward(user_input, item_num, user_bias, repeats=None, pred_slc=None)

      Forward the model by user

      :param user_input: user input tensor
      :type user_input: torch.Tensor
      :param item_num: user history interaction lens
      :type item_num: torch.Tensor
      :param repeats: the number of items to be evaluated
      :type repeats: int, optional
      :param pred_slc: continuous index which controls the current evaluation items,
                       if pred_slc is None, it will evaluate all items
      :type pred_slc: torch.Tensor, optional

      :returns: result
      :rtype: torch.Tensor



   .. py:method:: forward(user, item)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



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



.. py:class:: GCMC(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   GCMC is a model that incorporate graph autoencoders for recommendation.

   Graph autoencoders are comprised of:

   1) a graph encoder model :math:`Z = f(X; A)`, which take as input an :math:`N \times D` feature matrix X and
   a graph adjacency matrix A, and produce an :math:`N \times E` node embedding matrix
   :math:`Z = [z_1^T,..., z_N^T ]^T`;

   2) a pairwise decoder model :math:`\hat A = g(Z)`, which takes pairs of node embeddings :math:`(z_i, z_j)` and
   predicts respective entries :math:`\hat A_{ij}` in the adjacency matrix.

   Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
   and :math:`E` the embedding size.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: num_all


   .. py:attribute:: dropout_prob


   .. py:attribute:: sparse_feature


   .. py:attribute:: gcn_output_dim


   .. py:attribute:: dense_output_dim


   .. py:attribute:: n_class


   .. py:attribute:: num_basis_functions


   .. py:attribute:: input_dim


   .. py:attribute:: Graph


   .. py:attribute:: support


   .. py:attribute:: accum


   .. py:attribute:: GcEncoder


   .. py:attribute:: BiDecoder


   .. py:attribute:: loss_function


   .. py:method:: forward(user_X, item_X, user, item)


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



.. py:class:: ItemKNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   ItemKNN is a basic model that compute item similarity with the interaction matrix.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: k


   .. py:attribute:: shrink


   .. py:attribute:: interaction_matrix


   .. py:attribute:: pred_mat


   .. py:attribute:: fake_loss


   .. py:attribute:: other_parameter_name
      :value: ['w', 'pred_mat']



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



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



.. py:class:: LDiffRec(config, dataset)

   Bases: :py:obj:`hopwise.model.general_recommender.diffrec.DiffRec`


   L-DiffRec clusters items into groups, compresses the interaction vector over each group into a
   low-dimensional latent vector via a group-specific VAE, and conducts the forward and reverse
   diffusion processes in the latent space.


   .. py:attribute:: n_cate


   .. py:attribute:: reparam


   .. py:attribute:: ae_act_func


   .. py:attribute:: in_dims


   .. py:attribute:: out_dims


   .. py:attribute:: update_count
      :value: 0



   .. py:attribute:: update_count_vae
      :value: 0



   .. py:attribute:: lamda


   .. py:attribute:: anneal_cap


   .. py:attribute:: anneal_steps


   .. py:attribute:: vae_anneal_cap


   .. py:attribute:: vae_anneal_steps


   .. py:attribute:: autoencoder


   .. py:attribute:: latent_size


   .. py:attribute:: mlp


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



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



.. py:class:: LightGCN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   LightGCN is a GCN-based recommender model.

   LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
   collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
   propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
   learned at all layers as the final embedding.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: latent_dim


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: require_pow


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: norm_adj_matrix


   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]



   .. py:method:: forward()


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



.. py:class:: LINE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   LINE is a graph embedding model.

   We implement the model to train users and items embedding for recommendation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: order


   .. py:attribute:: second_order_loss_weight


   .. py:attribute:: interaction_feat


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: loss_fct


   .. py:attribute:: used_ids


   .. py:attribute:: random_list


   .. py:attribute:: random_pr
      :value: 0



   .. py:attribute:: random_list_length


   .. py:method:: sampler(key_ids)


   .. py:method:: random_num(num)


   .. py:method:: get_user_id_list()


   .. py:method:: forward(h, t)


   .. py:method:: context_forward(h, t, field)


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



.. py:class:: MacridVAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   MacridVAE is an item-based collaborative filtering model that learns disentangled representations from user
   behavior and simultaneously ranks all items for each user.

   We implement the model following the original author.


   .. py:attribute:: input_type


   .. py:attribute:: layers


   .. py:attribute:: embedding_size


   .. py:attribute:: drop_out


   .. py:attribute:: kfac


   .. py:attribute:: tau


   .. py:attribute:: nogb


   .. py:attribute:: anneal_cap


   .. py:attribute:: total_anneal_steps


   .. py:attribute:: regs


   .. py:attribute:: std


   .. py:attribute:: update
      :value: 0



   .. py:attribute:: encode_layer_dims


   .. py:attribute:: encoder


   .. py:attribute:: item_embedding


   .. py:attribute:: k_embedding


   .. py:attribute:: l2_loss


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: reg_loss()

      Calculate the L2 normalization loss of model parameters.
      Including embedding matrices and weight matrices of model.

      :returns: The L2 Loss tensor. shape of [1,]
      :rtype: loss(torch.FloatTensor)



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



.. py:class:: MultiDAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   MultiDAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

   We implement the the MultiDAE model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: layers


   .. py:attribute:: lat_dim


   .. py:attribute:: drop_out


   .. py:attribute:: encode_layer_dims


   .. py:attribute:: decode_layer_dims


   .. py:attribute:: encoder


   .. py:attribute:: decoder


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: forward(rating_matrix)


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



.. py:class:: MultiVAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   MultiVAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

   We implement the MultiVAE model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: layers


   .. py:attribute:: lat_dim


   .. py:attribute:: drop_out


   .. py:attribute:: anneal_cap


   .. py:attribute:: total_anneal_steps


   .. py:attribute:: update
      :value: 0



   .. py:attribute:: encode_layer_dims


   .. py:attribute:: decode_layer_dims


   .. py:attribute:: encoder


   .. py:attribute:: decoder


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix)


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



.. py:class:: NAIS(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NAIS is an attention network, which is capable of distinguishing which historical items
   in a user profile are more important for a prediction. We just implement the model following
   the original author with a pointwise training mode.

   .. note::

      instead of forming a minibatch as all training instances of a randomly sampled user which is
      mentioned in the original paper, we still train the model by a randomly sampled interactions.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: weight_size


   .. py:attribute:: algorithm


   .. py:attribute:: reg_weights


   .. py:attribute:: alpha


   .. py:attribute:: beta


   .. py:attribute:: split_to


   .. py:attribute:: pretrain_path


   .. py:attribute:: item_src_embedding


   .. py:attribute:: item_dst_embedding


   .. py:attribute:: bias


   .. py:attribute:: weight_layer


   .. py:attribute:: bceloss


   .. py:method:: _init_weights(module)

      Initialize the module's parameters

      .. note::

         It's a little different from the source code, because pytorch has no function to initialize
         the parameters by truncated normal distribution, so we replace it with xavier normal distribution



   .. py:method:: _load_pretrain()

      A simple implementation of loading pretrained parameters.



   .. py:method:: get_history_info(dataset)

      Get the user history interaction information

      :param dataset: train dataset
      :type dataset: DataSet

      :returns: (history_item_matrix, history_lens, mask_mat)
      :rtype: tuple



   .. py:method:: reg_loss()

      Calculate the reg loss for embedding layers and mlp layers

      :returns: reg loss
      :rtype: torch.Tensor



   .. py:method:: attention_mlp(inter, target)

      Layers of attention which support `prod` and `concat`

      :param inter: the embedding of history items
      :type inter: torch.Tensor
      :param target: the embedding of target items
      :type target: torch.Tensor

      :returns: the result of attention
      :rtype: torch.Tensor



   .. py:method:: mask_softmax(similarity, logits, bias, item_num, batch_mask_mat)

      Softmax the unmasked user history items and get the final output

      :param similarity: the similarity between the history items and target items
      :type similarity: torch.Tensor
      :param logits: the initial weights of the history items
      :type logits: torch.Tensor
      :param item_num: user history interaction lengths
      :type item_num: torch.Tensor
      :param bias: bias
      :type bias: torch.Tensor
      :param batch_mask_mat: the mask of user history interactions
      :type batch_mask_mat: torch.Tensor

      :returns: final output
      :rtype: torch.Tensor



   .. py:method:: softmax(similarity, logits, item_num, bias)

      Softmax the user history features and get the final output

      :param similarity: the similarity between the history items and target items
      :type similarity: torch.Tensor
      :param logits: the initial weights of the history items
      :type logits: torch.Tensor
      :param item_num: user history interaction lengths
      :type item_num: torch.Tensor
      :param bias: bias
      :type bias: torch.Tensor

      :returns: final output
      :rtype: torch.Tensor



   .. py:method:: inter_forward(user, item)

      Forward the model by interaction



   .. py:method:: user_forward(user_input, item_num, repeats=None, pred_slc=None)

      Forward the model by user

      :param user_input: user input tensor
      :type user_input: torch.Tensor
      :param item_num: user history interaction lens
      :type item_num: torch.Tensor
      :param repeats: the number of items to be evaluated
      :type repeats: int, optional
      :param pred_slc: continuous index which controls the current evaluation items,
                       if pred_slc is None, it will evaluate all items
      :type pred_slc: torch.Tensor, optional

      :returns: result
      :rtype: torch.Tensor



   .. py:method:: forward(user, item)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



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



.. py:class:: NCEPLRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   This is a abstract general recommender. All the general model should implement this class.
   The base general recommender class provide the basic dataset and parameters information.


   .. py:attribute:: input_type


   .. py:attribute:: dummy_param


   .. py:attribute:: user_embeddings


   .. py:attribute:: item_embeddings


   .. py:method:: forward()


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



.. py:class:: NCL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NCL is a neighborhood-enriched contrastive learning paradigm for graph collaborative filtering.
   Both structural and semantic neighbors are explicitly captured as contrastive learning objects.


   .. py:attribute:: input_type


   .. py:attribute:: latent_dim


   .. py:attribute:: n_layers


   .. py:attribute:: reg_weight


   .. py:attribute:: ssl_temp


   .. py:attribute:: ssl_reg


   .. py:attribute:: hyper_layers


   .. py:attribute:: alpha


   .. py:attribute:: proto_reg


   .. py:attribute:: k


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: norm_adj_matrix


   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:attribute:: user_centroids
      :value: None



   .. py:attribute:: user_2cluster
      :value: None



   .. py:attribute:: item_centroids
      :value: None



   .. py:attribute:: item_2cluster
      :value: None



   .. py:method:: e_step()


   .. py:method:: run_kmeans(x)

      Run K-means algorithm to get k clusters of the input tensor x



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]



   .. py:method:: forward()


   .. py:method:: ProtoNCE_loss(node_embedding, user, item)


   .. py:method:: ssl_layer_loss(current_embedding, previous_embedding, user, item)


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



.. py:class:: NeuMF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NeuMF is an neural network enhanced matrix factorization model.
   It replace the dot product to mlp for a more precise user-item interaction.

   .. note:: Our implementation only contains a rough pretraining function.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: mf_embedding_size


   .. py:attribute:: mlp_embedding_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: mf_train


   .. py:attribute:: mlp_train


   .. py:attribute:: use_pretrain


   .. py:attribute:: mf_pretrain_path


   .. py:attribute:: mlp_pretrain_path


   .. py:attribute:: user_mf_embedding


   .. py:attribute:: item_mf_embedding


   .. py:attribute:: user_mlp_embedding


   .. py:attribute:: item_mlp_embedding


   .. py:attribute:: mlp_layers


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: load_pretrain()

      A simple implementation of loading pretrained parameters.



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



   .. py:method:: dump_parameters()

      A simple implementation of dumping model parameters for pretrain.



.. py:class:: NGCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NGCF is a model that incorporate GNN for recommendation.
   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size_list


   .. py:attribute:: node_dropout


   .. py:attribute:: message_dropout


   .. py:attribute:: reg_weight


   .. py:attribute:: sparse_dropout


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: emb_dropout


   .. py:attribute:: GNNlayers


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: norm_adj_matrix


   .. py:attribute:: eye_matrix


   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)



   .. py:method:: forward()


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



.. py:class:: NNCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NNCF is an neural network enhanced matrix factorization model which also captures neighborhood information.
   We implement the NNCF model with three ways to process neighborhood information.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: interaction_matrix


   .. py:attribute:: ui_embedding_size


   .. py:attribute:: neigh_embedding_size


   .. py:attribute:: num_conv_kernel


   .. py:attribute:: conv_kernel_size


   .. py:attribute:: pool_kernel_size


   .. py:attribute:: mlp_hidden_size


   .. py:attribute:: neigh_num


   .. py:attribute:: neigh_info_method


   .. py:attribute:: resolution


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: user_neigh_embedding


   .. py:attribute:: item_neigh_embedding


   .. py:attribute:: user_conv


   .. py:attribute:: item_conv


   .. py:attribute:: mlp_layers


   .. py:attribute:: out_layer


   .. py:attribute:: dropout_layer


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: Max_ner(lst, max_ner)

      Unify embedding length of neighborhood information for efficiency consideration.
      Truncate the list if the length is larger than max_ner.
      Otherwise, pad it with 0.

      :param lst: The input list contains node's neighbors.
      :type lst: list
      :param max_ner: The number of neighbors we choose for each node.
      :type max_ner: int

      :returns: The list of a node's community neighbors.
      :rtype: list



   .. py:method:: get_community_member(partition, community_dict, node, kind)

      Find other nodes in the same community.
      e.g. If the node starts with letter "i",
      the other nodes start with letter "i" in the same community dict group are its community neighbors.

      :param partition: The input dict that contains the community each node belongs.
      :type partition: dict
      :param community_dict: The input dict that shows the nodes each community contains.
      :type community_dict: dict
      :param node: The id of the input node.
      :type node: int
      :param kind: The type of the input node.
      :type kind: char

      :returns: The list of a node's community neighbors.
      :rtype: list



   .. py:method:: prepare_vector_element(partition, relation, community_dict)

      Find the community neighbors of each node, i.e. I(u) and U(i).
      Then reset the id of nodes.

      :param partition: The input dict that contains the community each node belongs.
      :type partition: dict
      :param relation: The input list that contains the relationships of users and items.
      :type relation: list
      :param community_dict: The input dict that shows the nodes each community contains.
      :type community_dict: dict

      :returns: The list of nodes' community neighbors.
      :rtype: list



   .. py:method:: get_neigh_louvain()

      Get neighborhood information using louvain algorithm.
      First, change the id of node,
      for example, the id of user node "1" will be set to "u_1" in order to use louvain algorithm.
      Second, use louvain algorithm to seperate nodes into different communities.
      Finally, find the community neighbors of each node with the same type and reset the id of the nodes.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_knn()

      Get neighborhood information using knn algorithm.
      Find direct neighbors of each node, if the number of direct neighbors is less than neigh_num,
      add other similar neighbors using knn algorithm.
      Otherwise, select random top k direct neighbors, k equals to the number of neighbors.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_random()

      Get neighborhood information using random algorithm.
      Select random top k direct neighbors, k equals to the number of neighbors.

      :returns: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
      :rtype: torch.IntTensor



   .. py:method:: get_neigh_info(user, item)

      Get a batch of neighborhood embedding tensor according to input id.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor
      :param item: The input tensor that contains item's id, shape: [batch_size, ]
      :type item: torch.LongTensor

      :returns: The neighborhood embedding tensor of a batch of user, shape: [batch_size, neigh_embedding_size]
                torch.FloatTensor: The neighborhood embedding tensor of a batch of item, shape: [batch_size, neigh_embedding_size]
      :rtype: torch.FloatTensor



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



.. py:class:: Pop(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   Pop is an fundamental model that always recommend the most popular item.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: item_cnt


   .. py:attribute:: max_cnt
      :value: None



   .. py:attribute:: fake_loss


   .. py:attribute:: other_parameter_name
      :value: ['item_cnt', 'max_cnt']



   .. py:method:: forward()


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



.. py:class:: RaCT(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   RaCT is a collaborative filtering model which uses methods based on actor-critic reinforcement learning for training.

   We implement the RaCT model with only user dataloader.


   .. py:attribute:: input_type


   .. py:attribute:: layers


   .. py:attribute:: lat_dim


   .. py:attribute:: drop_out


   .. py:attribute:: anneal_cap


   .. py:attribute:: total_anneal_steps


   .. py:attribute:: update
      :value: 0



   .. py:attribute:: encode_layer_dims


   .. py:attribute:: decode_layer_dims


   .. py:attribute:: encoder


   .. py:attribute:: decoder


   .. py:attribute:: critic_layers


   .. py:attribute:: metrics_k


   .. py:attribute:: number_of_seen_items
      :value: 0



   .. py:attribute:: number_of_unseen_items
      :value: 0



   .. py:attribute:: critic_layer_dims


   .. py:attribute:: input_matrix
      :value: None



   .. py:attribute:: predict_matrix
      :value: None



   .. py:attribute:: true_matrix
      :value: None



   .. py:attribute:: critic_net


   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


   .. py:method:: mlp_layers(layer_dims)


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix)


   .. py:method:: calculate_actor_loss(interaction)


   .. py:method:: construct_critic_input(actor_loss)


   .. py:method:: construct_critic_layers(layer_dims)


   .. py:method:: calculate_ndcg(predict_matrix, true_matrix, input_matrix, k)


   .. py:method:: critic_forward(actor_loss)


   .. py:method:: calculate_critic_loss(interaction)


   .. py:method:: calculate_ac_loss(interaction)


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



.. py:class:: Random(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   Random is an fundamental model that recommends random items.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: fake_loss


   .. py:method:: forward()


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



.. py:class:: RecVAE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`, :py:obj:`hopwise.model.abstract_recommender.AutoEncoderMixin`


   Collaborative Denoising Auto-Encoder (RecVAE) is a recommendation model
   for top-N recommendation with implicit feedback.

   We implement the model following the original author


   .. py:attribute:: input_type


   .. py:attribute:: hidden_dim


   .. py:attribute:: latent_dim


   .. py:attribute:: dropout_prob


   .. py:attribute:: beta


   .. py:attribute:: mixture_weights


   .. py:attribute:: gamma


   .. py:attribute:: encoder


   .. py:attribute:: prior


   .. py:attribute:: decoder


   .. py:method:: reparameterize(mu, logvar)


   .. py:method:: forward(rating_matrix, dropout_prob)


   .. py:method:: calculate_loss(interaction, encoder_flag)

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



   .. py:method:: update_prior()


.. py:class:: SGL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SGL is a GCN-based recommender model.

   SGL supplements the classical supervised task of recommendation with an auxiliary
   self supervised task, which reinforces node representation learning via self-
   discrimination.Specifically,SGL generates multiple views of a node, maximizing the
   agreement between different views of the same node compared to that of other nodes.
   SGL devises three operators to generate the views — node dropout, edge dropout, and
   random walk — that change the graph structure in different manners.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: _user


   .. py:attribute:: _item


   .. py:attribute:: embed_dim


   .. py:attribute:: n_layers


   .. py:attribute:: type


   .. py:attribute:: drop_ratio


   .. py:attribute:: ssl_tau


   .. py:attribute:: reg_weight


   .. py:attribute:: ssl_weight


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: reg_loss


   .. py:attribute:: train_graph


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: graph_construction()

      Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.



   .. py:method:: rand_sample(high, size=None, replace=True)

      Randomly discard some points or edges.

      :param high: Upper limit of index value
      :type high: int
      :param size: Array size after sampling
      :type size: int

      :returns: Array index after sampling, shape: [size]
      :rtype: numpy.ndarray



   .. py:method:: create_adjust_matrix(is_sub: bool)

      Get the normalized interaction matrix of users and items.

      Construct the square matrix from the training data and normalize it
      using the laplace matrix.If it is a subgraph, it may be processed by
      node dropout or edge dropout.

      .. math::
          A_{hat} = D^{-0.5} \times A \times D^{-0.5}

      :returns: csr_matrix of the normalized interaction matrix.



   .. py:method:: csr2tensor(matrix: scipy.sparse.csr_matrix)

      Convert csr_matrix to tensor.

      :param matrix: Sparse matrix to be converted.
      :type matrix: scipy.csr_matrix

      :returns: Transformed sparse matrix.
      :rtype: torch.sparse.FloatTensor



   .. py:method:: forward(graph)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)

      Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

      :param user_emd: Ego embedding of all users after forwarding.
      :type user_emd: torch.Tensor
      :param item_emd: Ego embedding of all items after forwarding.
      :type item_emd: torch.Tensor
      :param user_list: List of the user.
      :type user_list: torch.Tensor
      :param pos_item_list: List of positive examples.
      :type pos_item_list: torch.Tensor
      :param neg_item_list: List of negative examples.
      :type neg_item_list: torch.Tensor

      :returns: Loss of BPR tasks and parameter regularization.
      :rtype: torch.Tensor



   .. py:method:: calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2)

      Calculate the loss of self-supervised tasks.

      :param user_list: List of the user.
      :type user_list: torch.Tensor
      :param pos_item_list: List of positive examples.
      :type pos_item_list: torch.Tensor
      :param user_sub1: Ego embedding of all users in the first subgraph after forwarding.
      :type user_sub1: torch.Tensor
      :param user_sub2: Ego embedding of all users in the second subgraph after forwarding.
      :type user_sub2: torch.Tensor
      :param item_sub1: Ego embedding of all items in the first subgraph after forwarding.
      :type item_sub1: torch.Tensor
      :param item_sub2: Ego embedding of all items in the second subgraph after forwarding.
      :type item_sub2: torch.Tensor

      :returns: Loss of self-supervised tasks.
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



   .. py:method:: train(mode: bool = True)

      Override train method of base class.The subgraph is reconstructed each time it is called.



.. py:class:: SimpleX(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SimpleX is a simple, unified collaborative filtering model.

   SimpleX presents a simple and easy-to-understand model. Its advantage lies
   in its loss function, which uses a larger number of negative samples and
   sets a threshold to filter out less informative samples, it also uses
   relative weights to control the balance of positive-sample loss
   and negative-sample loss.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: history_item_id


   .. py:attribute:: history_item_len


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: negative_weight


   .. py:attribute:: gamma


   .. py:attribute:: neg_seq_len


   .. py:attribute:: reg_weight


   .. py:attribute:: aggregator


   .. py:attribute:: history_len


   .. py:attribute:: user_emb


   .. py:attribute:: item_emb


   .. py:attribute:: UI_map


   .. py:attribute:: dropout


   .. py:attribute:: require_pow


   .. py:attribute:: reg_loss


   .. py:method:: get_UI_aggregation(user_e, history_item_e, history_len)

      Get the combined vector of user and historically interacted items

      :param user_e: User's feature vector, shape: [user_num, embedding_size]
      :type user_e: torch.Tensor
      :param history_item_e: History item's feature vector,
                             shape: [user_num, max_history_len, embedding_size]
      :type history_item_e: torch.Tensor
      :param history_len: User's history length, shape: [user_num]
      :type history_len: torch.Tensor

      :returns: Combined vector of user and item sequences, shape: [user_num, embedding_size]
      :rtype: torch.Tensor



   .. py:method:: get_cos(user_e, item_e)

      Get the cosine similarity between user and item

      :param user_e: User's feature vector, shape: [user_num, embedding_size]
      :type user_e: torch.Tensor
      :param item_e: Item's feature vector,
                     shape: [user_num, item_num, embedding_size]
      :type item_e: torch.Tensor

      :returns: Cosine similarity between user and item, shape: [user_num, item_num]
      :rtype: torch.Tensor



   .. py:method:: forward(user, pos_item, history_item, history_len, neg_item_seq)

      Get the loss

      :param user: User's id, shape: [user_num]
      :type user: torch.Tensor
      :param pos_item: Positive item's id, shape: [user_num]
      :type pos_item: torch.Tensor
      :param history_item: Id of historty item, shape: [user_num, max_history_len]
      :type history_item: torch.Tensor
      :param history_len: History item's length, shape: [user_num]
      :type history_len: torch.Tensor
      :param neg_item_seq: Negative item seq's id, shape: [user_num, neg_seq_len]
      :type neg_item_seq: torch.Tensor

      :returns: Loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_loss(interaction)

      Data processing and call function forward(), return loss

      To use SimpleX, a user must have a historical transaction record,
      a pos item and a sequence of neg items. Based on the hopwise
      framework, the data in the interaction object is ordered, so
      we can get the data quickly.



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



.. py:class:: SLIMElastic(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SLIMElastic is a sparse linear method for top-K recommendation, which learns
   a sparse aggregation coefficient matrix by solving an L1-norm and L2-norm
   regularized optimization problem.



   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: hide_item


   .. py:attribute:: alpha


   .. py:attribute:: l1_ratio


   .. py:attribute:: positive_only


   .. py:attribute:: dummy_param


   .. py:attribute:: interaction_matrix


   .. py:attribute:: item_similarity


   .. py:attribute:: other_parameter_name
      :value: ['interaction_matrix', 'item_similarity']



   .. py:method:: forward()


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



.. py:class:: SpectralCF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SpectralCF is a spectral convolution model that directly learns latent factors of users and items
   from the spectral domain for recommendation.

   The spectral convolution operation with C input channels and F filters is shown as the following:

   .. math::
       \left[\begin{array} {c} X_{new}^{u} \\
       X_{new}^{i} \end{array}\right]=\sigma\left(\left(U U^{\top}+U \Lambda U^{\top}\right)
       \left[\begin{array}{c} X^{u} \\
       X^{i} \end{array}\right] \Theta^{\prime}\right)

   where :math:`X_{new}^{u} \in R^{n_{users} \times F}` and :math:`X_{new}^{i} \in R^{n_{items} \times F}`
   denote convolution results learned with F filters from the spectral domain for users and items, respectively;
   :math:`\sigma` denotes the logistic sigmoid function.

   .. note::

      Our implementation is a improved version which is different from the original paper.
      For a better stability, we replace :math:`U U^T` with identity matrix :math:`I` and
      replace :math:`U \Lambda U^T` with laplace matrix :math:`L`.


   .. py:attribute:: input_type


   .. py:attribute:: n_layers


   .. py:attribute:: emb_dim


   .. py:attribute:: reg_weight


   .. py:attribute:: A_hat


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: filters


   .. py:attribute:: sigmoid


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_item_e']



   .. py:method:: get_ego_embeddings()

      Get the embedding of users and items and combine to an embedding matrix.

      :returns: Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)



   .. py:method:: forward()


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



