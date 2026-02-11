hopwise.model.loss
==================

.. py:module:: hopwise.model.loss

.. autoapi-nested-parse::

   hopwise.model.loss
   #######################
   Common Loss in recommender system



Classes
-------

.. autoapisummary::

   hopwise.model.loss.SSMLoss
   hopwise.model.loss.SimCELoss
   hopwise.model.loss.BPRLoss
   hopwise.model.loss.RegLoss
   hopwise.model.loss.EmbLoss
   hopwise.model.loss.EmbMarginLoss
   hopwise.model.loss.InnerProductLoss
   hopwise.model.loss.LogisticLoss


Module Contents
---------------

.. py:class:: SSMLoss(cosine_sim=True, temperature=1.0, eps=1e-07)

   Bases: :py:obj:`torch.nn.Module`


   Samples Softmax Loss (SSM) according to the implementation in


   .. py:attribute:: cosine_sim
      :value: True



   .. py:attribute:: temperature
      :value: 1.0



   .. py:attribute:: eps
      :value: 1e-07



   .. py:method:: forward(user_emb, pos_item_emb, neg_item_emb)


.. py:class:: SimCELoss(margin=5.0)

   Bases: :py:obj:`torch.nn.Module`


   Simplified Sampled Softmax Cross- Entropy Loss (SimCE),
   based on the implementation in https://arxiv.org/pdf/2406.16170


   .. py:attribute:: margin
      :value: 5.0



   .. py:method:: forward(user_emb, pos_item_emb, neg_item_emb)


.. py:class:: BPRLoss(gamma=1e-10)

   Bases: :py:obj:`torch.nn.Module`


   BPRLoss, based on Bayesian Personalized Ranking

   :param - gamma: Small value to avoid division by zero
   :type - gamma: float

   Shape:
       - Pos_score: (N)
       - Neg_score: (N), same shape as the Pos_score
       - Output: scalar.

   Examples::

       >>> loss = BPRLoss()
       >>> pos_score = torch.randn(3, requires_grad=True)
       >>> neg_score = torch.randn(3, requires_grad=True)
       >>> output = loss(pos_score, neg_score)
       >>> output.backward()


   .. py:attribute:: gamma
      :value: 1e-10



   .. py:method:: forward(pos_score, neg_score)


.. py:class:: RegLoss

   Bases: :py:obj:`torch.nn.Module`


   RegLoss, L2 regularization on model parameters


   .. py:method:: forward(parameters, reg_loss=None)


.. py:class:: EmbLoss(norm=2)

   Bases: :py:obj:`torch.nn.Module`


   EmbLoss, regularization on embeddings


   .. py:attribute:: norm
      :value: 2



   .. py:method:: forward(*embeddings, require_pow=False)


.. py:class:: EmbMarginLoss(power=2)

   Bases: :py:obj:`torch.nn.Module`


   EmbMarginLoss, regularization on embeddings


   .. py:attribute:: power
      :value: 2



   .. py:method:: forward(*embeddings)


.. py:class:: InnerProductLoss

   Bases: :py:obj:`torch.nn.Module`


   This is the inner-product loss used in CFKG for optimization.


   .. py:method:: forward(anchor, positive, negative)


.. py:class:: LogisticLoss

   Bases: :py:obj:`torch.nn.Module`


   This is the logistic loss


   .. py:attribute:: softplus


   .. py:method:: forward(positive_score, negative_score, pos_regularization=None, neg_regularization=None)


