hopwise.model.knowledge_aware_recommender.ripplenet
===================================================

.. py:module:: hopwise.model.knowledge_aware_recommender.ripplenet

.. autoapi-nested-parse::

   RippleNet
   #####################################################
   Reference:
       Hongwei Wang et al. "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems."
       in CIKM 2018.



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.ripplenet.RippleNet


Module Contents
---------------

.. py:class:: RippleNet(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   RippleNet is an knowledge enhanced matrix factorization model.
   The original interaction matrix of :math:`n_{users} \times n_{items}`
   and related knowledge graph is set as model input,
   we carefully design the data interface and use ripple set to train and test efficiently.
   We just implement the model following the original author with a pointwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_weight


   .. py:attribute:: reg_weight


   .. py:attribute:: n_hop


   .. py:attribute:: n_memory


   .. py:attribute:: interaction_matrix


   .. py:attribute:: kg


   .. py:attribute:: user_dict


   .. py:attribute:: ripple_set


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: transform_matrix


   .. py:attribute:: softmax


   .. py:attribute:: sigmoid


   .. py:attribute:: rec_loss


   .. py:attribute:: l2_loss


   .. py:attribute:: loss


   .. py:attribute:: other_parameter_name
      :value: ['ripple_set']



   .. py:method:: _build_ripple_set()

      Get the normalized interaction matrix of users and items according to A_values.
      Get the ripple hop-wise ripple set for every user, w.r.t. their interaction history

      :returns: ripple_set (dict)



   .. py:method:: forward(interaction)


   .. py:method:: _key_addressing()

      Conduct reasoning for specific item and user ripple set

      :returns: list of torch.cuda.FloatTensor n_hop * [batch_size, embedding_size]
      :rtype: o_list (dict -> torch.cuda.FloatTensor)



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



   .. py:method:: _key_addressing_full()

      Conduct reasoning for specific item and user ripple set

      :returns:

                list of torch.cuda.FloatTensor
                    n_hop * [batch_size, n_item, embedding_size]
      :rtype: o_list (dict -> torch.cuda.FloatTensor)



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



