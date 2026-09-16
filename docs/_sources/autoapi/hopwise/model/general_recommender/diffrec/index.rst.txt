hopwise.model.general_recommender.diffrec
=========================================

.. py:module:: hopwise.model.general_recommender.diffrec

.. autoapi-nested-parse::

   DiffRec
   ################################################
   Reference:
       Wenjie Wang et al. "Diffusion Recommender Model." in SIGIR 2023.

   Reference code:
       https://github.com/YiyanXu/DiffRec



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.diffrec.ModelMeanType
   hopwise.model.general_recommender.diffrec.DNN
   hopwise.model.general_recommender.diffrec.DiffRec


Functions
---------

.. autoapisummary::

   hopwise.model.general_recommender.diffrec.betas_from_linear_variance
   hopwise.model.general_recommender.diffrec.betas_for_alpha_bar
   hopwise.model.general_recommender.diffrec.normal_kl
   hopwise.model.general_recommender.diffrec.mean_flat
   hopwise.model.general_recommender.diffrec.timestep_embedding


Module Contents
---------------

.. py:class:: ModelMeanType

   Bases: :py:obj:`enum.Enum`


   Generic enumeration.

   Derive from this class to define new enumerations.


   .. py:attribute:: START_X


   .. py:attribute:: EPSILON


.. py:class:: DNN(dims: list, emb_size: int, time_type='cat', act_func='tanh', norm=False, dropout=0.5)

   Bases: :py:obj:`torch.nn.Module`


   A deep neural network for the reverse diffusion preocess.


   .. py:attribute:: dims


   .. py:attribute:: time_type
      :value: 'cat'



   .. py:attribute:: time_emb_dim


   .. py:attribute:: norm
      :value: False



   .. py:attribute:: emb_layer


   .. py:attribute:: mlp_layers


   .. py:attribute:: drop


   .. py:method:: forward(x, timesteps)


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



.. py:function:: betas_from_linear_variance(steps, variance, max_beta=0.999)

.. py:function:: betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999)

   Create a beta schedule that discretizes the given alpha_t_bar function,
   which defines the cumulative product of (1-beta) over time from t = [0,1].

   :param num_diffusion_timesteps: the number of betas to produce.
   :type num_diffusion_timesteps: int
   :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                     produces the cumulative product of (1-beta) up to that
                     part of the diffusion process.
   :type alpha_bar: Callable
   :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
   :type max_beta: int

   :returns: a 1-D array of beta values.
   :rtype: np.ndarray


.. py:function:: normal_kl(mean1, logvar1, mean2, logvar2)

   Compute the KL divergence between two gaussians.

   Shapes are automatically broadcasted, so batches can be compared to
   scalars, among other use cases.


.. py:function:: mean_flat(tensor)

   Take the mean over all non-batch dimensions.


.. py:function:: timestep_embedding(timesteps, dim, max_period=10000)

   Create sinusoidal timestep embeddings.

   :param timesteps: a 1-D Tensor of N indices, one per batch element.
                     These may be fractional. (N,)
   :param dim: the dimension of the output.
   :param max_period: controls the minimum frequency of the embeddings.
   :return: an [N x dim] Tensor of positional embeddings.


