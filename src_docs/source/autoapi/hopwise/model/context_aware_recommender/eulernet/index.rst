hopwise.model.context_aware_recommender.eulernet
================================================

.. py:module:: hopwise.model.context_aware_recommender.eulernet

.. autoapi-nested-parse::

   EulerNet
   ################################################
   Reference:
       Zhen Tian et al. "EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction." in SIGIR 2023.

   Reference code:
       https://github.com/chenyuwuxin/EulerNet



Classes
-------

.. autoapisummary::

   hopwise.model.context_aware_recommender.eulernet.EulerNet
   hopwise.model.context_aware_recommender.eulernet.EulerInteractionLayer


Module Contents
---------------

.. py:class:: EulerNet(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ContextRecommender`


   EulerNet is a context-based recommendation model.
   It can adaptively learn the arbitrary-order feature interactions in a complex vector space
   by conducting space mapping according to Euler's formula. Meanwhile, it can jointly capture
   the explicit and implicit feature interactions in a unified model architecture.


   .. py:attribute:: Euler_interaction_layers


   .. py:attribute:: mu


   .. py:attribute:: reg


   .. py:attribute:: reg_weight


   .. py:attribute:: sigmoid


   .. py:attribute:: reg_loss


   .. py:attribute:: loss


   .. py:method:: _init_other_weights(module)


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



   .. py:method:: RegularLoss(weight)


.. py:class:: EulerInteractionLayer(config, inshape, outshape)

   Bases: :py:obj:`torch.nn.Module`


   Euler interaction layer is the core component of EulerNet,
   which enables the adaptive learning of explicit feature interactions. An Euler
   interaction layer performs the feature interaction under the complex space one time,
   taking as input a complex representation and outputting a transformed complex representation.


   .. py:attribute:: feature_dim


   .. py:attribute:: apply_norm


   .. py:attribute:: inter_orders


   .. py:attribute:: im


   .. py:attribute:: bias_lam


   .. py:attribute:: bias_theta


   .. py:attribute:: drop_ex


   .. py:attribute:: drop_im


   .. py:attribute:: norm_r


   .. py:attribute:: norm_p


   .. py:method:: forward(complex_features)


