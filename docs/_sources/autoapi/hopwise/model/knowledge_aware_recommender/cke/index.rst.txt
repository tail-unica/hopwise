hopwise.model.knowledge_aware_recommender.cke
=============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.cke

.. autoapi-nested-parse::

   CKE
   ##################################################
   Reference:
       Fuzheng Zhang et al. "Collaborative Knowledge Base Embedding for Recommender Systems." in SIGKDD 2016.



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.cke.CKE


Module Contents
---------------

.. py:class:: CKE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   CKE is a knowledge-based recommendation model, it can incorporate KG and other information such as corresponding
   images to enrich the representation of items for item recommendations.

   .. note::

      In the original paper, CKE used structural knowledge, textual knowledge and visual knowledge. In our
      implementation, we only used structural knowledge. Meanwhile, the version we implemented uses a simpler
      regular way which can get almost the same result (even better) as the original regular way.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: reg_weights


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: trans_w


   .. py:attribute:: rec_loss


   .. py:attribute:: kg_loss


   .. py:attribute:: reg_loss


   .. py:method:: _get_kg_embedding(h, r, pos_t, neg_t)


   .. py:method:: forward(user, item)


   .. py:method:: _get_rec_loss(user_e, pos_e, neg_e)


   .. py:method:: _get_kg_loss(h_e, r_e, pos_e, neg_e)


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



