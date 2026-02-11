hopwise.model.sequential_recommender.sasrec
===========================================

.. py:module:: hopwise.model.sequential_recommender.sasrec

.. autoapi-nested-parse::

   SASRec
   ################################################

   Reference:
       Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

   Reference:
       https://github.com/kang205/SASRec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.sasrec.SASRec


Module Contents
---------------

.. py:class:: SASRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   SASRec is the first sequential recommender based on self-attentive mechanism.

   .. note::

      In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
      by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
      using Fully Connected Layer to implement the PFFN.


   .. py:attribute:: n_layers


   .. py:attribute:: n_heads


   .. py:attribute:: hidden_size


   .. py:attribute:: inner_size


   .. py:attribute:: hidden_dropout_prob


   .. py:attribute:: attn_dropout_prob


   .. py:attribute:: hidden_act


   .. py:attribute:: layer_norm_eps


   .. py:attribute:: initializer_range


   .. py:attribute:: loss_type


   .. py:attribute:: item_embedding


   .. py:attribute:: position_embedding


   .. py:attribute:: trm_encoder


   .. py:attribute:: LayerNorm


   .. py:attribute:: dropout


   .. py:method:: _init_weights(module)

      Initialize the weights



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



