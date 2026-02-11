hopwise.model.general_recommender.ldiffrec
==========================================

.. py:module:: hopwise.model.general_recommender.ldiffrec

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

   hopwise.model.general_recommender.ldiffrec.AutoEncoder
   hopwise.model.general_recommender.ldiffrec.LDiffRec


Functions
---------

.. autoapisummary::

   hopwise.model.general_recommender.ldiffrec.compute_loss


Module Contents
---------------

.. py:class:: AutoEncoder(item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1)

   Bases: :py:obj:`torch.nn.Module`


   Guassian Diffusion for large-scale recommendation.


   .. py:attribute:: item_emb


   .. py:attribute:: n_cate


   .. py:attribute:: in_dims


   .. py:attribute:: out_dims


   .. py:attribute:: act_func


   .. py:attribute:: n_item


   .. py:attribute:: reparam
      :value: True



   .. py:attribute:: dropout


   .. py:method:: Encode(batch)


   .. py:method:: reparamterization(mu, logvar)


   .. py:method:: Decode(batch)


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



.. py:function:: compute_loss(recon_x, x)

