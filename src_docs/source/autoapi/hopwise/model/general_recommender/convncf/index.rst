hopwise.model.general_recommender.convncf
=========================================

.. py:module:: hopwise.model.general_recommender.convncf

.. autoapi-nested-parse::

   ConvNCF
   ################################################
   Reference:
       Xiangnan He et al. "Outer Product-based Neural Collaborative Filtering." in IJCAI 2018.

   Reference code:
       https://github.com/duxy-me/ConvNCF



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.convncf.ConvNCFBPRLoss
   hopwise.model.general_recommender.convncf.ConvNCF


Module Contents
---------------

.. py:class:: ConvNCFBPRLoss

   Bases: :py:obj:`torch.nn.Module`


   ConvNCFBPRLoss, based on Bayesian Personalized Ranking,

   Shape:
       - Pos_score: (N)
       - Neg_score: (N), same shape as the Pos_score
       - Output: scalar.

   Examples::

       >>> loss = ConvNCFBPRLoss()
       >>> pos_score = torch.randn(3, requires_grad=True)
       >>> neg_score = torch.randn(3, requires_grad=True)
       >>> output = loss(pos_score, neg_score)
       >>> output.backward()


   .. py:method:: forward(pos_score, neg_score)


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



