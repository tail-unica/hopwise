hopwise.model.sequential_recommender.gru4reccpr
===============================================

.. py:module:: hopwise.model.sequential_recommender.gru4reccpr

.. autoapi-nested-parse::

   GRU4Rec + Softmax-CPR
   ################################################

   Reference:
       Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.
       Haw-Shiuan Chang, Nikhil Agarwal, and Andrew McCallum "To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders" in WSDM 2024




Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.gru4reccpr.GRU4RecCPR


Functions
---------

.. autoapisummary::

   hopwise.model.sequential_recommender.gru4reccpr.gelu


Module Contents
---------------

.. py:function:: gelu(x)

.. py:class:: GRU4RecCPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   GRU4Rec is a model that incorporate RNN for recommendation.

   .. note::

      Regarding the innovation of this article,we can only achieve the data augmentation mentioned
      in the paper and directly output the embedding of the item,
      in order that the generation method we used is common to other sequential models.


   .. py:attribute:: hidden_size


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: n_facet_all


   .. py:attribute:: n_facet


   .. py:attribute:: n_facet_window


   .. py:attribute:: n_facet_hidden


   .. py:attribute:: n_facet_MLP


   .. py:attribute:: n_facet_context


   .. py:attribute:: n_facet_reranker


   .. py:attribute:: n_facet_emb


   .. py:attribute:: softmax_nonlinear
      :value: 'None'



   .. py:attribute:: use_out_emb


   .. py:attribute:: only_compute_loss
      :value: True



   .. py:attribute:: dense


   .. py:attribute:: n_embd


   .. py:attribute:: use_proj_bias


   .. py:attribute:: weight_mode


   .. py:attribute:: context_norm


   .. py:attribute:: post_remove_context


   .. py:attribute:: reranker_merging_mode


   .. py:attribute:: partition_merging_mode


   .. py:attribute:: reranker_CAN_NUM


   .. py:attribute:: candidates_from_previous_reranker
      :value: True



   .. py:attribute:: MLP_linear


   .. py:attribute:: project_arr


   .. py:attribute:: project_emb


   .. py:attribute:: c
      :value: 123



   .. py:attribute:: emb_dropout


   .. py:attribute:: gru_layers


   .. py:attribute:: item_embedding


   .. py:method:: _init_weights(module)


   .. py:method:: forward(item_seq, item_seq_len)


   .. py:method:: get_facet_emb(input_emb, i)


   .. py:method:: calculate_loss_prob(interaction, only_compute_prob=False)


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



