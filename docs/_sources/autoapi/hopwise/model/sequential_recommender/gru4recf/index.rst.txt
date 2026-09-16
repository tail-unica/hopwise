hopwise.model.sequential_recommender.gru4recf
=============================================

.. py:module:: hopwise.model.sequential_recommender.gru4recf

.. autoapi-nested-parse::

   GRU4RecF
   ################################################

   Reference:
       Balázs Hidasi et al. "Parallel Recurrent Neural Network Architectures for
       Feature-rich Session-based Recommendations." in RecSys 2016.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.gru4recf.GRU4RecF


Module Contents
---------------

.. py:class:: GRU4RecF(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   In the original paper, the authors proposed several architectures. We compared 3 different
   architectures:

       (1)  Concatenate item input and feature input and use single RNN,

       (2)  Concatenate outputs from two different RNNs,

       (3)  Weighted sum of outputs from two different RNNs.

   We implemented the optimal parallel version(2), which uses different RNNs to
   encode items and features respectively and concatenates the two subparts'
   outputs as the final output. The different RNN encoders are trained simultaneously.


   .. py:attribute:: embedding_size


   .. py:attribute:: hidden_size


   .. py:attribute:: num_layers


   .. py:attribute:: dropout_prob


   .. py:attribute:: selected_features


   .. py:attribute:: pooling_mode


   .. py:attribute:: device


   .. py:attribute:: num_feature_field


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: feature_embed_layer


   .. py:attribute:: item_gru_layers


   .. py:attribute:: feature_gru_layers


   .. py:attribute:: dense_layer


   .. py:attribute:: dropout


   .. py:attribute:: other_parameter_name
      :value: ['feature_embed_layer']



   .. py:method:: forward(item_seq, item_seq_len)


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



