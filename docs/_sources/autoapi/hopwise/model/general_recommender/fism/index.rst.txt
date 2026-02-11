hopwise.model.general_recommender.fism
======================================

.. py:module:: hopwise.model.general_recommender.fism

.. autoapi-nested-parse::

   FISM
   #######################################
   Reference:
       S. Kabbur et al. "FISM: Factored item similarity models for top-n recommender systems" in KDD 2013

   Reference code:
       https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.fism.FISM


Module Contents
---------------

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



