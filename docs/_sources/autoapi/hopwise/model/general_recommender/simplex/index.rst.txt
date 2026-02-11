hopwise.model.general_recommender.simplex
=========================================

.. py:module:: hopwise.model.general_recommender.simplex

.. autoapi-nested-parse::

   SimpleX
   ################################################

   Reference:
       Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

   Reference code:
       https://github.com/xue-pai/TwoToweRS



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.simplex.SimpleX


Module Contents
---------------

.. py:class:: SimpleX(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SimpleX is a simple, unified collaborative filtering model.

   SimpleX presents a simple and easy-to-understand model. Its advantage lies
   in its loss function, which uses a larger number of negative samples and
   sets a threshold to filter out less informative samples, it also uses
   relative weights to control the balance of positive-sample loss
   and negative-sample loss.

   We implement the model following the original author with a pairwise training mode.


   .. py:attribute:: input_type


   .. py:attribute:: history_item_id


   .. py:attribute:: history_item_len


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: negative_weight


   .. py:attribute:: gamma


   .. py:attribute:: neg_seq_len


   .. py:attribute:: reg_weight


   .. py:attribute:: aggregator


   .. py:attribute:: history_len


   .. py:attribute:: user_emb


   .. py:attribute:: item_emb


   .. py:attribute:: UI_map


   .. py:attribute:: dropout


   .. py:attribute:: require_pow


   .. py:attribute:: reg_loss


   .. py:method:: get_UI_aggregation(user_e, history_item_e, history_len)

      Get the combined vector of user and historically interacted items

      :param user_e: User's feature vector, shape: [user_num, embedding_size]
      :type user_e: torch.Tensor
      :param history_item_e: History item's feature vector,
                             shape: [user_num, max_history_len, embedding_size]
      :type history_item_e: torch.Tensor
      :param history_len: User's history length, shape: [user_num]
      :type history_len: torch.Tensor

      :returns: Combined vector of user and item sequences, shape: [user_num, embedding_size]
      :rtype: torch.Tensor



   .. py:method:: get_cos(user_e, item_e)

      Get the cosine similarity between user and item

      :param user_e: User's feature vector, shape: [user_num, embedding_size]
      :type user_e: torch.Tensor
      :param item_e: Item's feature vector,
                     shape: [user_num, item_num, embedding_size]
      :type item_e: torch.Tensor

      :returns: Cosine similarity between user and item, shape: [user_num, item_num]
      :rtype: torch.Tensor



   .. py:method:: forward(user, pos_item, history_item, history_len, neg_item_seq)

      Get the loss

      :param user: User's id, shape: [user_num]
      :type user: torch.Tensor
      :param pos_item: Positive item's id, shape: [user_num]
      :type pos_item: torch.Tensor
      :param history_item: Id of historty item, shape: [user_num, max_history_len]
      :type history_item: torch.Tensor
      :param history_len: History item's length, shape: [user_num]
      :type history_len: torch.Tensor
      :param neg_item_seq: Negative item seq's id, shape: [user_num, neg_seq_len]
      :type neg_item_seq: torch.Tensor

      :returns: Loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_loss(interaction)

      Data processing and call function forward(), return loss

      To use SimpleX, a user must have a historical transaction record,
      a pos item and a sequence of neg items. Based on the hopwise
      framework, the data in the interaction object is ordered, so
      we can get the data quickly.



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



