hopwise.model.knowledge_aware_recommender.pgpr
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.pgpr

.. autoapi-nested-parse::

   PGPR
   ##################################################
   Reference:
       Xian et al. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." in SIGIR 2019.

   Reference code:
       https://github.com/orcax/PGPR



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.pgpr.PGPR
   hopwise.model.knowledge_aware_recommender.pgpr.KGState


Module Contents
---------------

.. py:class:: PGPR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`, :py:obj:`hopwise.model.abstract_recommender.ExplainableRecommender`


   This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
   The base knowledge-based recommender class provide the basic dataset and parameters information.


   .. py:attribute:: input_type


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


   .. py:attribute:: fix_scores_sorting_bug


   .. py:attribute:: ui_relation_id


   .. py:attribute:: graph_dict


   .. py:attribute:: positives


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


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



   .. py:method:: select_action(batch_state, batch_act_mask)


   .. py:method:: update()


   .. py:method:: forward(inputs)


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


   .. py:method:: collect_scores(users, paths, probs)


.. py:class:: KGState(embedding_size, history_len)

   .. py:attribute:: embedding_size


   .. py:attribute:: history_len


   .. py:method:: __call__(user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed)


