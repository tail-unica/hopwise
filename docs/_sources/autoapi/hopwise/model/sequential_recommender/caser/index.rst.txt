hopwise.model.sequential_recommender.caser
==========================================

.. py:module:: hopwise.model.sequential_recommender.caser

.. autoapi-nested-parse::

   Caser
   ################################################

   Reference:
       Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.

   Reference code:
       https://github.com/graytowne/caser_pytorch



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.caser.Caser


Module Contents
---------------

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



