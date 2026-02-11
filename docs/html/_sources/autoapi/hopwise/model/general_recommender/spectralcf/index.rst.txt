hopwise.model.general_recommender.spectralcf
============================================

.. py:module:: hopwise.model.general_recommender.spectralcf

.. autoapi-nested-parse::

   SpectralCF
   ################################################

   Reference:
       Lei Zheng et al. "Spectral collaborative filtering." in RecSys 2018.

   Reference code:
       https://github.com/lzheng21/SpectralCF



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.spectralcf.SpectralCF


Module Contents
---------------

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



