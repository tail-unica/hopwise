hopwise.model.knowledge_aware_recommender.tprec
===============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.tprec

.. autoapi-nested-parse::

   TPRec
   ##################################################
   Reference: Time-aware Path Reasoning on Knowledge Graph for Recommendation (https://arxiv.org/pdf/2108.02634)



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.tprec.TPRec
   hopwise.model.knowledge_aware_recommender.tprec.KGState


Module Contents
---------------

.. py:class:: TPRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`, :py:obj:`hopwise.model.abstract_recommender.ExplainableRecommender`


   TPRec

   1. Train TransE embeddings and preprocess it according to the preprocess embedding notebook
   2. Run TPRec with 'pretrain' train stage set
   3. Run TPRec with 'policy' train stage set



   .. py:attribute:: input_type


   .. py:attribute:: config


   .. py:attribute:: user_num


   .. py:attribute:: device


   .. py:attribute:: topk


   .. py:attribute:: state_history


   .. py:attribute:: max_acts


   .. py:attribute:: gamma


   .. py:attribute:: action_dropout


   .. py:attribute:: hidden_sizes


   .. py:attribute:: act_dim


   .. py:attribute:: max_num_nodes


   .. py:attribute:: weight_factor


   .. py:attribute:: path_pattern


   .. py:attribute:: beam_search_hop


   .. py:attribute:: train_stage


   .. py:attribute:: margin


   .. py:attribute:: n_clusters


   .. py:attribute:: fix_scores_sorting_bug


   .. py:attribute:: ui_relation


   .. py:attribute:: ui_relation_id


   .. py:attribute:: graph_dict


   .. py:attribute:: positives


   .. py:attribute:: pretrained_weights


   .. py:attribute:: uc_weight


   .. py:attribute:: timenum


   .. py:attribute:: ui2label_dict


   .. py:attribute:: embedding_size


   .. py:attribute:: state_gen


   .. py:attribute:: l1


   .. py:attribute:: l2


   .. py:attribute:: actor


   .. py:attribute:: critic


   .. py:attribute:: self_loop_embedding


   .. py:attribute:: rid2relation


   .. py:attribute:: node_type2emb


   .. py:attribute:: u_p_scales


   .. py:attribute:: patterns
      :value: []



   .. py:attribute:: _batch_path
      :value: None



   .. py:attribute:: _batch_curr_actions
      :value: None



   .. py:attribute:: _batch_curr_state
      :value: None



   .. py:attribute:: _batch_curr_reward
      :value: None



   .. py:attribute:: _done
      :value: False



   .. py:attribute:: saved_actions
      :value: []



   .. py:attribute:: rewards
      :value: []



   .. py:attribute:: entropy
      :value: []



   .. py:attribute:: rng


   .. py:class:: SavedAction

      Bases: :py:obj:`tuple`


      .. py:attribute:: log_prob


      .. py:attribute:: value



   .. py:method:: expand_paths(path_constraint)

      Expand the path constraint by replacing the ui_relation with multiple clusters
      like ui_relation0 to ui_relation_n with n being the number of clusters.

      :param path_constraint in the form of [rel1:
      :param rel2:
      :param rel3]:



   .. py:method:: _get_pretrained_weights()


   .. py:method:: select_action(batch_state, batch_act_mask)


   .. py:method:: update()


   .. py:method:: forward(inputs)


   .. py:method:: _get_transe_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_transe_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: calculate_loss_transe(interaction)


   .. py:method:: forward_transe(user, relation, item)


   .. py:method:: predict_transe(interaction)


   .. py:method:: full_sort_predict_transe(interaction)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: _has_pattern(path)


   .. py:method:: _get_next_node_type(current_node_type, relation_id)


   .. py:method:: reset(user)


   .. py:method:: _batch_get_actions(batch_path, done)


   .. py:method:: _get_actions(path, done)


   .. py:method:: _batch_get_state(batch_path)


   .. py:method:: _get_state(path)


   .. py:method:: _batch_get_reward(batch_path)


   .. py:method:: _get_reward(path)


   .. py:method:: _is_done()


   .. py:method:: batch_step(batch_act_idx)


   .. py:method:: batch_action_mask(dropout)


   .. py:method:: _batch_acts_to_masks(batch_acts)


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



   .. py:method:: _build_interacted_matrix(temporal_weight)


   .. py:method:: explain(interaction)

      Support function used for case study.

      :param interaction: test interaction data

      :returns: explanation results with columns: "user", "item", "score", "path"
      :rtype: pd.Dataframe



   .. py:method:: decode_path(path)

      Decode the path into a string. Path decoding is specific to each model.

      :param path: The path data.
      :type path: list

      :returns: The decoded path string.
      :rtype: str



   .. py:method:: beam_search(users)


   .. py:method:: collect_scores(users, paths, probs, interacted_matrix)


.. py:class:: KGState(embedding_size, history_len)

   .. py:attribute:: embedding_size


   .. py:attribute:: history_len


   .. py:method:: __call__(user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed)


