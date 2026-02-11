hopwise.model.sequential_recommender.nextitnet
==============================================

.. py:module:: hopwise.model.sequential_recommender.nextitnet

.. autoapi-nested-parse::

   NextItNet
   ################################################

   Reference:
       Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.

   Reference code:
       - https://github.com/fajieyuan/nextitnet
       - https://github.com/initlisk/nextitnet_pytorch



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.nextitnet.NextItNet
   hopwise.model.sequential_recommender.nextitnet.ResidualBlock_a
   hopwise.model.sequential_recommender.nextitnet.ResidualBlock_b


Module Contents
---------------

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



.. py:class:: ResidualBlock_a(in_channel, out_channel, kernel_size=3, dilation=None)

   Bases: :py:obj:`torch.nn.Module`


   Residual block (a) in the paper


   .. py:attribute:: ln1


   .. py:attribute:: conv1


   .. py:attribute:: ln2


   .. py:attribute:: conv2


   .. py:attribute:: ln3


   .. py:attribute:: conv3


   .. py:attribute:: dilation
      :value: None



   .. py:attribute:: kernel_size
      :value: 3



   .. py:method:: forward(x)


   .. py:method:: conv_pad(x, dilation)

      Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
      trick for the 1D dilated convolution to prevent the network from seeing the future items.
      Also the One-dimensional transformation is completed in this function.



.. py:class:: ResidualBlock_b(in_channel, out_channel, kernel_size=3, dilation=None)

   Bases: :py:obj:`torch.nn.Module`


   Residual block (b) in the paper


   .. py:attribute:: conv1


   .. py:attribute:: ln1


   .. py:attribute:: conv2


   .. py:attribute:: ln2


   .. py:attribute:: dilation
      :value: None



   .. py:attribute:: kernel_size
      :value: 3



   .. py:method:: forward(x)


   .. py:method:: conv_pad(x, dilation)

      Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
      trick for the 1D dilated convolution to prevent the network from seeing the future items.
      Also the One-dimensional transformation is completed in this function.



