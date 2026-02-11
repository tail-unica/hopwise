hopwise.model.knowledge_aware_recommender.kglrr
===============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.kglrr


Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.kglrr.GraphAttentionLayer
   hopwise.model.knowledge_aware_recommender.kglrr.KGEncoder
   hopwise.model.knowledge_aware_recommender.kglrr.KGLRR
   hopwise.model.knowledge_aware_recommender.kglrr.GAT


Module Contents
---------------

.. py:class:: GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: dropout


   .. py:attribute:: in_features


   .. py:attribute:: out_features


   .. py:attribute:: alpha


   .. py:attribute:: concat
      :value: True



   .. py:attribute:: W


   .. py:attribute:: a


   .. py:attribute:: fc


   .. py:attribute:: leakyrelu


   .. py:method:: forward_relation(item_embs, entity_embs, relations, adj)


   .. py:method:: forward(item_embs, entity_embs, adj)


   .. py:method:: _prepare_cat(Wh, We)


.. py:class:: KGEncoder(config, dataset, kg_dataset)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: user_history_matrix


   .. py:attribute:: maxhis


   .. py:attribute:: kgcn


   .. py:attribute:: dropout


   .. py:attribute:: keep_prob


   .. py:attribute:: A_split


   .. py:attribute:: device


   .. py:attribute:: latent_dim


   .. py:attribute:: n_layers


   .. py:attribute:: max_entities_per_user


   .. py:attribute:: kg_dataset


   .. py:attribute:: gat


   .. py:attribute:: inter_feat


   .. py:attribute:: num_users


   .. py:attribute:: num_items


   .. py:attribute:: config


   .. py:method:: __init_weight(dataset)


   .. py:method:: get_kg_dict(item_num)


   .. py:method:: computer()


   .. py:method:: __dropout_x(x, keep_prob)


   .. py:method:: __dropout(keep_prob)


   .. py:method:: cal_item_embedding_from_kg(kg: dict)


   .. py:method:: cal_item_embedding_gat(kg: dict)


   .. py:method:: cal_item_embedding_rgat(kg: dict)


.. py:class:: KGLRR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGLRR: Reinforced logical reasoning over KGs for interpretable recommendation system


   .. py:attribute:: input_type


   .. py:attribute:: kg_dataset


   .. py:attribute:: encoder


   .. py:attribute:: latent_dim


   .. py:attribute:: r_logic


   .. py:attribute:: r_length


   .. py:attribute:: layers


   .. py:attribute:: sim_scale


   .. py:attribute:: loss_sum


   .. py:attribute:: l2s_weight


   .. py:attribute:: is_explain


   .. py:attribute:: num_items


   .. py:attribute:: bceloss


   .. py:method:: _init_weights()


   .. py:method:: logic_or(vector1, vector2, train=False)


   .. py:method:: logic_and(vector1, vector2, train=False)


   .. py:method:: logic_regularizer(train: bool, check_list: list, constraint, constraint_valid)


   .. py:method:: similarity(vector1, vector2, sigmoid=True)


   .. py:method:: uniform_size(vector1, vector2, train=False)


   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: explain(users, history, items)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users, n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: predict_or_and(users, pos, neg, history)


   .. py:method:: calculate_loss(interaction)

      Calculates the total loss by combining:
      - BCE Loss (rloss)
      - Entropy Loss (tloss)
      - L2 Loss (l2loss)



   .. py:method:: triple_loss(TItemScore, FItemScore)


   .. py:method:: l2_loss(users, pos, neg, history)


   .. py:method:: check(check_list)

      Logs the shape and contents of tensors in the provided check_list.

      Each element in check_list is expected to be a tuple where the first item
      is a string (label) and the second item is a tensor. For each tuple, this
      function converts the tensor to a NumPy array after detaching it from the
      computation graph and moving it to CPU, then logs the label, shape and
      array contents with a threshold of 20 elements for display.

      :param check_list: List of (label, tensor) pairs to be logged for inspection.
      :type check_list: list of tuple



   .. py:method:: forward(print_check: bool, return_pred: bool, *args, **kwards)


.. py:class:: GAT(nfeat, nhid, dropout, alpha)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: dropout


   .. py:attribute:: layer


   .. py:method:: forward(item_embs, entity_embs, adj)


   .. py:method:: forward_relation(item_embs, entity_embs, w_r, adj)


