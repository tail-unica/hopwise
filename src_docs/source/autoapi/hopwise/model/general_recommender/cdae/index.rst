hopwise.model.general_recommender.cdae
======================================

.. py:module:: hopwise.model.general_recommender.cdae

.. autoapi-nested-parse::

   CDAE
   ################################################
   Reference:
       Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. In WSDM 2016.

   Reference code:
       https://github.com/jasonyaw/CDAE



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.cdae.CDAE


Module Contents
---------------

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



