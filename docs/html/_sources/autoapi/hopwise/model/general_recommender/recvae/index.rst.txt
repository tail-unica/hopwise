hopwise.model.general_recommender.recvae
========================================

.. py:module:: hopwise.model.general_recommender.recvae

.. autoapi-nested-parse::

   RecVAE
   ################################################
   Reference:
       Shenbin, Ilya, et al. "RecVAE: A new variational autoencoder for Top-N recommendations with implicit feedback." In WSDM 2020.

   Reference code:
       https://github.com/ilya-shenbin/RecVAE



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.recvae.CompositePrior
   hopwise.model.general_recommender.recvae.Encoder
   hopwise.model.general_recommender.recvae.RecVAE


Functions
---------

.. autoapisummary::

   hopwise.model.general_recommender.recvae.swish
   hopwise.model.general_recommender.recvae.log_norm_pdf


Module Contents
---------------

.. py:function:: swish(x)

   Swish activation function:

   .. math::
       \text{Swish}(x) = \frac{x}{1 + \exp(-x)}


.. py:function:: log_norm_pdf(x, mu, logvar)

.. py:class:: CompositePrior(hidden_dim, latent_dim, input_dim, mixture_weights)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: mixture_weights


   .. py:attribute:: mu_prior


   .. py:attribute:: logvar_prior


   .. py:attribute:: logvar_uniform_prior


   .. py:attribute:: encoder_old


   .. py:method:: forward(x, z)


.. py:class:: Encoder(hidden_dim, latent_dim, input_dim, eps=0.1)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: fc1


   .. py:attribute:: ln1


   .. py:attribute:: fc2


   .. py:attribute:: ln2


   .. py:attribute:: fc3


   .. py:attribute:: ln3


   .. py:attribute:: fc4


   .. py:attribute:: ln4


   .. py:attribute:: fc5


   .. py:attribute:: ln5


   .. py:attribute:: fc_mu


   .. py:attribute:: fc_logvar


   .. py:method:: forward(x, dropout_prob)


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


