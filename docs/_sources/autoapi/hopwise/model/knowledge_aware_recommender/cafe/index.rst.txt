hopwise.model.knowledge_aware_recommender.cafe
==============================================

.. py:module:: hopwise.model.knowledge_aware_recommender.cafe

.. autoapi-nested-parse::

   CAFE
   ##################################################
   Reference:
       Xian et al. "CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation." in CIKM 2020.

   Reference code:
       https://github.com/orcax/CAFE



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.cafe.CAFE
   hopwise.model.knowledge_aware_recommender.cafe.SymbolicNetwork
   hopwise.model.knowledge_aware_recommender.cafe.RelationModule
   hopwise.model.knowledge_aware_recommender.cafe.DeepRelationModule
   hopwise.model.knowledge_aware_recommender.cafe.ReplayMemory
   hopwise.model.knowledge_aware_recommender.cafe.KGMask
   hopwise.model.knowledge_aware_recommender.cafe.MetaProgramExecutor
   hopwise.model.knowledge_aware_recommender.cafe.NeuralProgramLayout
   hopwise.model.knowledge_aware_recommender.cafe.TreeNode


Module Contents
---------------

.. py:class:: CAFE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   CAFE is a knowledge-aware recommender system that uses symbolic reasoning
   over a knowledge graph to explain recommendations.

   .. note:: Assumes that each relation corresponds to a unique pair of entity types. e.g. ui-relation -> (user, item)


   .. py:attribute:: input_type


   .. py:attribute:: dataset


   .. py:attribute:: device


   .. py:attribute:: load_embeddings


   .. py:attribute:: raw_metapaths


   .. py:attribute:: rank_weight


   .. py:attribute:: deep_module


   .. py:attribute:: use_dropout


   .. py:attribute:: topk_candidates


   .. py:attribute:: sample_size


   .. py:attribute:: topk_paths


   .. py:attribute:: path_max_user_trials


   .. py:attribute:: ui_relation


   .. py:attribute:: ui_relation_id


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: topk_user_items


   .. py:attribute:: embedding_size


   .. py:attribute:: embeddings


   .. py:attribute:: rid2relation


   .. py:attribute:: relation2rid


   .. py:attribute:: positives


   .. py:attribute:: graph_dict


   .. py:attribute:: memory_size
      :value: 10000



   .. py:attribute:: replay_memory


   .. py:attribute:: relation_info


   .. py:attribute:: metapaths
      :value: []



   .. py:attribute:: mpath_ids


   .. py:attribute:: model


   .. py:attribute:: rng


   .. py:method:: _compute_top_items()


   .. py:method:: _get_batch_by_user(users)


   .. py:method:: _rev_rel(rel)


   .. py:method:: fast_sample_path_with_target(mpath_id, user_id, target_id, num_paths, sample_size=100)

      Sample one path given source and target, using BFS from both sides.

      :returns: List of entity ids forming the path.
      :rtype: list



   .. py:method:: count_paths_with_target(mpath_id, user_id, target_id, sample_size=50)

      This is an approx count, not exact.



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



   .. py:method:: convert_path_relations(paths)


   .. py:method:: explain(interaction)

      Support function used for case study.

      :param interaction: test interaction data

      :returns: explanation results with columns: "user", "item", "score", "path"
      :rtype: pd.Dataframe



   .. py:method:: decode_path(path)


   .. py:method:: _infer_paths(users, kg_mask)


   .. py:method:: _estimate_path_count(users)


   .. py:method:: run_program(users, path_counts, predicted_paths)


   .. py:method:: create_heuristic_program(metapaths, predicted_paths, path_counts)


.. py:class:: SymbolicNetwork(relation_info, relation2rid, embeddings, embedding_size, deep_module, use_dropout, n_items, device)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: embedding


   .. py:attribute:: embedding_size


   .. py:attribute:: n_items


   .. py:attribute:: device


   .. py:attribute:: relation2rid


   .. py:attribute:: ce_loss


   .. py:method:: _create_modules(relation_info, use_deep=False, use_dropout=True)

      Create module for each relation.



   .. py:method:: _get_modules(metapath)

      Get list of modules by metapath.



   .. py:method:: _forward(modules, uids)


   .. py:method:: forward(metapath, pos_paths, neg_pids)

      Compute loss.

      :param metapath: list of relations, e.g. [USER, (r1, e1),..., (r_n, e_n)].
      :param pos_paths: a LongTensor of node ids, with size [bs, len(metapath)],
                        e.g. each path contains [u, e1,..., e_n].
      :param neg_pids: a LongTensor of negative product ids.

      :returns: sum of log probabilities of given target node ids, with size [bs, ].
      :rtype: logprobs



   .. py:method:: forward_simple(metapath, uids, pids)


   .. py:method:: infer_direct(metapath, uid, pids)


   .. py:method:: infer_with_path(metapath, uid, kg_mask, excluded_pids, topk_paths)

      Reasoning paths over kg.



.. py:class:: RelationModule(embedding_size, relation_info)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: name


   .. py:attribute:: eh_name


   .. py:attribute:: et_name


   .. py:attribute:: fc1


   .. py:attribute:: bn1


   .. py:attribute:: fc2


   .. py:attribute:: dropout


   .. py:method:: forward(inputs)

      Compute log probability of output entity.
      :param x: a FloatTensor of size [bs, input_size].

      :returns: FloatTensor of log probability of size [bs, output_size].



.. py:class:: DeepRelationModule(embedding_size, relation_info, use_dropout)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: name


   .. py:attribute:: eh_name


   .. py:attribute:: et_name


   .. py:attribute:: fc1


   .. py:attribute:: bn1


   .. py:attribute:: fc2


   .. py:attribute:: bn2


   .. py:attribute:: fc3


   .. py:method:: forward(inputs)


.. py:class:: ReplayMemory(memory_size=5000)

   .. py:attribute:: memory_size
      :value: 5000



   .. py:attribute:: memory
      :value: []



   .. py:method:: add(data)


   .. py:method:: sample()


   .. py:method:: __len__()


.. py:class:: KGMask(kg, ui_relation_id)

   .. py:attribute:: kg


   .. py:attribute:: ui_relation_id


   .. py:method:: _get_next_node_type(current_node_type, relation_id)


   .. py:method:: get_ids(eh, eh_ids, relation)


   .. py:method:: get_mask(eh, eh_ids, relation)


   .. py:method:: __call__(eh, eh_ids, relation)


.. py:class:: MetaProgramExecutor(symbolic_model, random_generator, device, kg_mask, relation2rid)

   This implements the profile-guided reasoning algorithm.


   .. py:attribute:: symbolic_model


   .. py:attribute:: kg_mask


   .. py:attribute:: device


   .. py:attribute:: relation2rid


   .. py:attribute:: rng


   .. py:method:: _get_module(relation)


   .. py:method:: execute(program, uid, excluded_pids=None, adaptive_topk=False, manual_topk=5)

      Execute the program to generate node representations and real nodes.
      :param program: an instance of MetaProgram.
      :param uid: user ID (integer).
      :param excluded_pids: list of item IDs (list).



   .. py:method:: collect_results(program)


.. py:class:: NeuralProgramLayout(metapaths)

   This refers to the layout tree in the paper.


   .. py:attribute:: mp2id


   .. py:attribute:: root


   .. py:method:: update_by_path_count(path_count)

      Update sample size of each node by expected number of paths.
      :param path_count: dict with key=mpid, value=int



   .. py:method:: print_postorder(hide_branch=True)


.. py:class:: TreeNode(level, entity, relation)

   .. py:attribute:: level


   .. py:attribute:: entity


   .. py:attribute:: relation


   .. py:attribute:: parent
      :value: None



   .. py:attribute:: children


   .. py:attribute:: sample_size
      :value: 0



   .. py:attribute:: data


   .. py:method:: has_parent()


   .. py:method:: has_children()


   .. py:method:: get_children()


   .. py:method:: __str__()


