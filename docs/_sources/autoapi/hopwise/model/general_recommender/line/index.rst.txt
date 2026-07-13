hopwise.model.general_recommender.line
======================================

.. py:module:: hopwise.model.general_recommender.line

.. autoapi-nested-parse::

   LINE
   ################################################
   Reference:
       Jian Tang et al. "LINE: Large-scale Information Network Embedding." in WWW 2015.

   Reference code:
       https://github.com/shenweichen/GraphEmbedding



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.line.NegSamplingLoss
   hopwise.model.general_recommender.line.LINE


Module Contents
---------------

.. py:class:: NegSamplingLoss

   Bases: :py:obj:`torch.nn.Module`


   .. py:method:: forward(sign, score)


.. py:class:: LINE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   LINE is a graph embedding model.

   We implement the model to train users and items embedding for recommendation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: order


   .. py:attribute:: second_order_loss_weight


   .. py:attribute:: interaction_feat


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: loss_fct


   .. py:attribute:: used_ids


   .. py:attribute:: random_list


   .. py:attribute:: random_pr
      :value: 0



   .. py:attribute:: random_list_length


   .. py:method:: sampler(key_ids)


   .. py:method:: random_num(num)


   .. py:method:: get_user_id_list()


   .. py:method:: forward(h, t)


   .. py:method:: context_forward(h, t, field)


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



