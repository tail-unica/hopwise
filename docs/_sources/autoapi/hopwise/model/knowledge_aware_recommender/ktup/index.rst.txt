hopwise.model.knowledge_aware_recommender.ktup
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.ktup

.. autoapi-nested-parse::

   KTUP
   ##################################################
   Reference:
       Yixin Cao et al. "Unifying Knowledge Graph Learning and Recommendation:Towards a Better Understanding
       of User Preferences." in WWW 2019.

   Reference code:
       https://github.com/TaoMiner/joint-kg-recommender



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.ktup.KTUP


Functions
---------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.ktup.orthogonalLoss
   hopwise.model.knowledge_aware_recommender.ktup.alignLoss


Module Contents
---------------

.. py:class:: KTUP(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KTUP is a knowledge-based recommendation model. It adopts the strategy of multi-task learning to jointly learn
   recommendation and KG-related tasks, with the goal of understanding the reasons that a user interacts with an item.
   This method utilizes an attention mechanism to combine all preferences into a single-vector representation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: L1_flag


   .. py:attribute:: use_st_gumbel


   .. py:attribute:: kg_weight


   .. py:attribute:: align_weight


   .. py:attribute:: margin


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: pref_embedding


   .. py:attribute:: pref_norm_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: relation_norm_embedding


   .. py:attribute:: rec_loss


   .. py:attribute:: kg_loss


   .. py:attribute:: reg_loss


   .. py:method:: _masked_softmax(logits)


   .. py:method:: convert_to_one_hot(indices, num_classes)

      :param indices: A vector containing indices,
                      whose size is (batch_size,).
      :type indices: Variable
      :param num_classes: The number of classes, which would be
                          the second dimension of the resulting one-hot matrix.
      :type num_classes: Variable

      :returns: The one-hot matrix of size (batch_size, num_classes).
      :rtype: torch.Tensor



   .. py:method:: st_gumbel_softmax(logits, temperature=1.0)

      Return the result of Straight-Through Gumbel-Softmax Estimation.
      It approximates the discrete sampling via Gumbel-Softmax trick
      and applies the biased ST estimator.
      In the forward propagation, it emits the discrete one-hot result,
      and in the backward propagation it approximates the categorical
      distribution via smooth Gumbel-Softmax distribution.

      :param logits: A un-normalized probability values,
                     which has the size (batch_size, num_classes)
      :type logits: Variable
      :param temperature: A temperature parameter. The higher
                          the value is, the smoother the distribution is.
      :type temperature: float

      :returns: The sampled output, which has the property explained above.
      :rtype: torch.Tensor



   .. py:method:: _get_preferences(user_e, item_e, use_st_gumbel=False)


   .. py:method:: _transH_projection(original, norm)
      :staticmethod:



   .. py:method:: _get_score(h_e, r_e, t_e)


   .. py:method:: forward(user, item)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.

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



.. py:function:: orthogonalLoss(rel_embeddings, norm_embeddings)

.. py:function:: alignLoss(emb1, emb2, L1_flag=False)

