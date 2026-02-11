hopwise.model.general_recommender.nais
======================================

.. py:module:: hopwise.model.general_recommender.nais

.. autoapi-nested-parse::

   NAIS
   ######################################
   Reference:
       Xiangnan He et al. "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE 2018.

   Reference code:
       https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.nais.NAIS


Module Contents
---------------

.. py:class:: NAIS(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   NAIS is an attention network, which is capable of distinguishing which historical items
   in a user profile are more important for a prediction. We just implement the model following
   the original author with a pointwise training mode.

   .. note::

      instead of forming a minibatch as all training instances of a randomly sampled user which is
      mentioned in the original paper, we still train the model by a randomly sampled interactions.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: weight_size


   .. py:attribute:: algorithm


   .. py:attribute:: reg_weights


   .. py:attribute:: alpha


   .. py:attribute:: beta


   .. py:attribute:: split_to


   .. py:attribute:: pretrain_path


   .. py:attribute:: item_src_embedding


   .. py:attribute:: item_dst_embedding


   .. py:attribute:: bias


   .. py:attribute:: weight_layer


   .. py:attribute:: bceloss


   .. py:method:: _init_weights(module)

      Initialize the module's parameters

      .. note::

         It's a little different from the source code, because pytorch has no function to initialize
         the parameters by truncated normal distribution, so we replace it with xavier normal distribution



   .. py:method:: _load_pretrain()

      A simple implementation of loading pretrained parameters.



   .. py:method:: get_history_info(dataset)

      Get the user history interaction information

      :param dataset: train dataset
      :type dataset: DataSet

      :returns: (history_item_matrix, history_lens, mask_mat)
      :rtype: tuple



   .. py:method:: reg_loss()

      Calculate the reg loss for embedding layers and mlp layers

      :returns: reg loss
      :rtype: torch.Tensor



   .. py:method:: attention_mlp(inter, target)

      Layers of attention which support `prod` and `concat`

      :param inter: the embedding of history items
      :type inter: torch.Tensor
      :param target: the embedding of target items
      :type target: torch.Tensor

      :returns: the result of attention
      :rtype: torch.Tensor



   .. py:method:: mask_softmax(similarity, logits, bias, item_num, batch_mask_mat)

      Softmax the unmasked user history items and get the final output

      :param similarity: the similarity between the history items and target items
      :type similarity: torch.Tensor
      :param logits: the initial weights of the history items
      :type logits: torch.Tensor
      :param item_num: user history interaction lengths
      :type item_num: torch.Tensor
      :param bias: bias
      :type bias: torch.Tensor
      :param batch_mask_mat: the mask of user history interactions
      :type batch_mask_mat: torch.Tensor

      :returns: final output
      :rtype: torch.Tensor



   .. py:method:: softmax(similarity, logits, item_num, bias)

      Softmax the user history features and get the final output

      :param similarity: the similarity between the history items and target items
      :type similarity: torch.Tensor
      :param logits: the initial weights of the history items
      :type logits: torch.Tensor
      :param item_num: user history interaction lengths
      :type item_num: torch.Tensor
      :param bias: bias
      :type bias: torch.Tensor

      :returns: final output
      :rtype: torch.Tensor



   .. py:method:: inter_forward(user, item)

      Forward the model by interaction



   .. py:method:: user_forward(user_input, item_num, repeats=None, pred_slc=None)

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



