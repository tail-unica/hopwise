hopwise.model.sequential_recommender.transrec
=============================================

.. py:module:: hopwise.model.sequential_recommender.transrec

.. autoapi-nested-parse::

   TransRec
   ################################################

   Reference:
       Ruining He et al. "Translation-based Recommendation." In RecSys 2017.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.transrec.TransRec


Module Contents
---------------

.. py:class:: TransRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   TransRec is translation-based model for sequential recommendation.
   It assumes that the `prev. item` + `user`  = `next item`.
   We use the Euclidean Distance to calculate the similarity in this implementation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_users


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: bias


   .. py:attribute:: T


   .. py:attribute:: bpr_loss


   .. py:attribute:: emb_loss


   .. py:attribute:: reg_loss


   .. py:method:: _l2_distance(x, y)


   .. py:method:: gather_last_items(item_seq, gather_index)

      Gathers the last_item at the specific positions over a minibatch



   .. py:method:: forward(user, item_seq, item_seq_len)


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



