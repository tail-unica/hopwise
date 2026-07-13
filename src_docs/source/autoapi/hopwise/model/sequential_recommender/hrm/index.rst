hopwise.model.sequential_recommender.hrm
========================================

.. py:module:: hopwise.model.sequential_recommender.hrm

.. autoapi-nested-parse::

   HRM
   ################################################

   Reference:
       Pengfei Wang et al. "Learning Hierarchical Representation Model for Next Basket Recommendation." in SIGIR 2015.

   Reference code:
       https://github.com/wubinzzu/NeuRec



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.hrm.HRM


Module Contents
---------------

.. py:class:: HRM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   HRM can well capture both sequential behavior and users’ general taste by involving transaction and
   user representations in prediction.

   HRM user max- & average- pooling as a good helper.


   .. py:attribute:: n_user


   .. py:attribute:: device


   .. py:attribute:: embedding_size


   .. py:attribute:: pooling_type_layer_1


   .. py:attribute:: pooling_type_layer_2


   .. py:attribute:: high_order


   .. py:attribute:: reg_weight


   .. py:attribute:: dropout_prob


   .. py:attribute:: item_embedding


   .. py:attribute:: user_embedding


   .. py:attribute:: dropout


   .. py:attribute:: loss_type


   .. py:method:: inverse_seq_item(seq_item, seq_item_len)

      Inverse the seq_item, like this
      [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]



   .. py:method:: _init_weights(module)


   .. py:method:: forward(seq_item, user, seq_item_len)


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



