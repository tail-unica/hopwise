hopwise.model.general_recommender.neumf
=======================================

.. py:module:: hopwise.model.general_recommender.neumf

.. autoapi-nested-parse::

   NeuMF
   ################################################
   Reference:
       Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.neumf.NeuMF


Module Contents
---------------

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



