hopwise.model.context_aware_recommender.afm
===========================================

.. py:module:: hopwise.model.context_aware_recommender.afm

.. autoapi-nested-parse::

   AFM
   ################################################
   Reference:
       Jun Xiao et al. "Attentional Factorization Machines: Learning the Weight of Feature Interactions via
       Attention Networks" in IJCAI 2017.



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.afm.AFM


Module Contents
---------------

.. py:class:: AFM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   AFM is a attention based FM model that predict the final score with the attention of input feature.


   .. py:attribute:: attention_size


   .. py:attribute:: dropout_prob


   .. py:attribute:: reg_weight


   .. py:attribute:: num_pair
      :value: 0.0



   .. py:attribute:: attlayer


   .. py:attribute:: p


   .. py:attribute:: dropout_layer


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: build_cross(feat_emb)

      Build the cross feature columns of feature columns

      :param feat_emb: input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
      :type feat_emb: torch.FloatTensor

      :returns:     - torch.FloatTensor: Left part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
                    - torch.FloatTensor: Right part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
      :rtype: tuple



   .. py:method:: afm_layer(infeature)

      Get the attention-based feature interaction score

      :param infeature: input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
      :type infeature: torch.FloatTensor

      :returns: Result of score. shape of [batch_size, 1].
      :rtype: torch.FloatTensor



   .. py:method:: forward(interaction)


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



