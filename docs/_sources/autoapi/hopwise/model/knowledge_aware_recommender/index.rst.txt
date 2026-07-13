hopwise.model.knowledge_aware_recommender
=========================================

.. py:module:: hopwise.model.knowledge_aware_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/knowledge_aware_recommender/cafe/index
   /autoapi/hopwise/model/knowledge_aware_recommender/cfkg/index
   /autoapi/hopwise/model/knowledge_aware_recommender/cke/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kgat/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kgcn/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kgin/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kglrr/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kgnnls/index
   /autoapi/hopwise/model/knowledge_aware_recommender/kgrec/index
   /autoapi/hopwise/model/knowledge_aware_recommender/ktup/index
   /autoapi/hopwise/model/knowledge_aware_recommender/mcclk/index
   /autoapi/hopwise/model/knowledge_aware_recommender/mkr/index
   /autoapi/hopwise/model/knowledge_aware_recommender/pgpr/index
   /autoapi/hopwise/model/knowledge_aware_recommender/ripplenet/index
   /autoapi/hopwise/model/knowledge_aware_recommender/tprec/index


Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_aware_recommender.CAFE
   hopwise.model.knowledge_aware_recommender.CFKG
   hopwise.model.knowledge_aware_recommender.CKE
   hopwise.model.knowledge_aware_recommender.KGAT
   hopwise.model.knowledge_aware_recommender.KGCN
   hopwise.model.knowledge_aware_recommender.KGIN
   hopwise.model.knowledge_aware_recommender.KGLRR
   hopwise.model.knowledge_aware_recommender.KGNNLS
   hopwise.model.knowledge_aware_recommender.KGRec
   hopwise.model.knowledge_aware_recommender.KTUP
   hopwise.model.knowledge_aware_recommender.MCCLK
   hopwise.model.knowledge_aware_recommender.MKR
   hopwise.model.knowledge_aware_recommender.PGPR
   hopwise.model.knowledge_aware_recommender.RippleNet
   hopwise.model.knowledge_aware_recommender.TPRec


Package Contents
----------------

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


.. py:class:: CFKG(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   CFKG is a knowledge-based recommendation model, it combines knowledge graph and the user-item interaction
   graph to a new graph. In this graph, user, item and related attribute are viewed as entities, and the interaction
   between user and item and the link between item and attribute are viewed as relations. It define a new score
   function as follows:

   .. math::
       d (u_i + r_{buy}, v_j)

   .. note::

      In the original paper, CFKG puts recommender data (u-i interaction) and knowledge data (h-r-t) together
      for sampling and mix them for training. In this version, we sample recommender data
      and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: rec_loss


   .. py:method:: forward(user, item)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: _get_score(h_e, t_e, r_e)


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



.. py:class:: KGAT(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
   graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
   items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.


   .. py:attribute:: input_type


   .. py:attribute:: ckg


   .. py:attribute:: all_hs


   .. py:attribute:: all_ts


   .. py:attribute:: all_rs


   .. py:attribute:: matrix_size


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: layers


   .. py:attribute:: aggregator_type


   .. py:attribute:: mess_dropout


   .. py:attribute:: reg_weight


   .. py:attribute:: A_in


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: trans_w


   .. py:attribute:: aggregator_layers


   .. py:attribute:: tanh


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:attribute:: other_parameter_name
      :value: ['restore_user_e', 'restore_entity_e']



   .. py:method:: init_graph()

      Get the initial attention matrix through the collaborative knowledge graph

      :returns: Sparse tensor of the attention matrix
      :rtype: torch.sparse.FloatTensor



   .. py:method:: _get_ego_embeddings()


   .. py:method:: forward()


   .. py:method:: _get_kg_embedding(h, r, pos_t, neg_t)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: generate_transE_score(hs, ts, r)

      Calculating scores for triples in KG.

      :param hs: head entities
      :type hs: torch.Tensor
      :param ts: tail entities
      :type ts: torch.Tensor
      :param r: the relation id between hs and ts
      :type r: int

      :returns: the scores of (hs, r, ts)
      :rtype: torch.Tensor



   .. py:method:: update_attentive_A()

      Update the attention matrix using the updated embedding matrix



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



.. py:class:: KGCN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGCN is a knowledge-based recommendation model that captures inter-item relatedness effectively by mining their
   associated attributes on the KG. To automatically discover both high-order structure information and semantic
   information of the KG, we treat KG as an undirected graph and sample from the neighbors for each entity in the KG
   as their receptive field, then combine neighborhood information with bias when calculating the representation of a
   given entity.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_iter


   .. py:attribute:: aggregator_class


   .. py:attribute:: reg_weight


   .. py:attribute:: neighbor_sample_size


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: softmax


   .. py:attribute:: linear_layers


   .. py:attribute:: ReLU


   .. py:attribute:: Tanh


   .. py:attribute:: bce_loss


   .. py:attribute:: l2_loss


   .. py:attribute:: other_parameter_name
      :value: ['adj_entity', 'adj_relation']



   .. py:method:: construct_adj(kg_graph)

      Get neighbors and corresponding relations for each entity in the KG.

      :param kg_graph: an undirected graph
      :type kg_graph: scipy.sparse.coo_matrix

      :returns:

                    - adj_entity(torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                      shape: [n_entities, neighbor_sample_size]
                    - adj_relation(torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                      shape: [n_entities, neighbor_sample_size]
      :rtype: tuple



   .. py:method:: get_neighbors(items)

      Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

      :param items: The input tensor that contains item's id, shape: [batch_size, ]
      :type items: torch.LongTensor

      :returns:

                    - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                      dimensions of entities: {[batch_size, 1],
                      [batch_size, n_neighbor],
                      [batch_size, n_neighbor^2],
                      ...,
                      [batch_size, n_neighbor^n_iter]}
                    - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                      entities. Relations have the same shape as entities.
      :rtype: tuple



   .. py:method:: mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

      Mix neighbor vectors on user-specific graph.

      :param neighbor_vectors: The embeddings of neighbor entities(items),
                               shape: [batch_size, -1, neighbor_sample_size, embedding_size]
      :type neighbor_vectors: torch.FloatTensor
      :param neighbor_relations: The embeddings of neighbor relations,
                                 shape: [batch_size, -1, neighbor_sample_size, embedding_size]
      :type neighbor_relations: torch.FloatTensor
      :param user_embeddings: The embeddings of users, shape: [batch_size, embedding_size]
      :type user_embeddings: torch.FloatTensor

      :returns: The neighbors aggregated embeddings,
                shape: [batch_size, -1, embedding_size]
      :rtype: neighbors_aggregated(torch.FloatTensor)



   .. py:method:: aggregate(user_embeddings, entities, relations)

      For each item, aggregate the entity representation and its neighborhood representation into a single vector.

      :param user_embeddings: The embeddings of users, shape: [batch_size, embedding_size]
      :type user_embeddings: torch.FloatTensor
      :param entities: entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                       dimensions of entities: {[batch_size, 1],
                       [batch_size, n_neighbor],
                       [batch_size, n_neighbor^2],
                       ...,
                       [batch_size, n_neighbor^n_iter]}
      :type entities: list
      :param relations: relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                        relations have the same shape as entities.
      :type relations: list

      :returns: The embeddings of items, shape: [batch_size, embedding_size]
      :rtype: item_embeddings(torch.FloatTensor)



   .. py:method:: forward(user, item)


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



.. py:class:: KGIN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGIN is a knowledge-aware recommendation model. It combines knowledge graph and the user-item interaction
   graph to a new graph called collaborative knowledge graph (CKG). This model explores intents behind a user-item
   interaction by using auxiliary item knowledge.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: n_factors


   .. py:attribute:: context_hops


   .. py:attribute:: node_dropout_rate


   .. py:attribute:: mess_dropout_rate


   .. py:attribute:: ind


   .. py:attribute:: sim_decay


   .. py:attribute:: reg_weight


   .. py:attribute:: temperature


   .. py:attribute:: interact_mat


   .. py:attribute:: kg_graph


   .. py:attribute:: n_nodes


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: latent_embedding


   .. py:attribute:: gcn


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data of KG.

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


.. py:class:: KGNNLS(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGNN-LS is a knowledge-based recommendation model.
   KGNN-LS transforms the knowledge graph into a user-specific weighted graph and then apply a graph neural network to
   compute personalized item embeddings. To provide better inductive bias, KGNN-LS relies on label smoothness
   assumption, which posits that adjacent items in the knowledge graph are likely to have similar user relevance
   labels/scores. Label smoothness provides regularization over the edge weights and it is equivalent  to a label
   propagation scheme on a graph.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: neighbor_sample_size


   .. py:attribute:: aggregator_class


   .. py:attribute:: n_iter


   .. py:attribute:: reg_weight


   .. py:attribute:: ls_weight


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: interaction_table


   .. py:attribute:: softmax


   .. py:attribute:: linear_layers


   .. py:attribute:: ReLU


   .. py:attribute:: Tanh


   .. py:attribute:: bce_loss


   .. py:attribute:: l2_loss


   .. py:attribute:: other_parameter_name
      :value: ['adj_entity', 'adj_relation']



   .. py:method:: get_interaction_table(user_id, item_id, y)

      Get interaction_table that is used for fetching user-item interaction label in LS regularization.

      :param user_id: the user id in user-item interactions, shape: [n_interactions, 1]
      :type user_id: torch.Tensor
      :param item_id: the item id in user-item interactions, shape: [n_interactions, 1]
      :type item_id: torch.Tensor
      :param y: the label in user-item interactions, shape: [n_interactions, 1]
      :type y: torch.Tensor

      :returns:     - interaction_table(dict): key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
                    - offset(int): The offset that is used for calculating the key(index) in interaction_table
      :rtype: tuple



   .. py:method:: sample_neg_interaction(pos_interaction_table, offset)

      Sample neg_interaction to construct train data.

      :param pos_interaction_table: the interaction_table that only contains pos_interaction.
      :type pos_interaction_table: dict
      :param offset: The offset that is used for calculating the key(index) in interaction_table
      :type offset: int

      :returns: key: user_id * 10^offset + item_id; value: y_{user_id, item_id}
      :rtype: interaction_table(dict)



   .. py:method:: construct_adj(kg_graph)

      Get neighbors and corresponding relations for each entity in the KG.

      :param kg_graph: an undirected graph
      :type kg_graph: scipy.sparse.coo_matrix

      :returns:

                    - adj_entity (torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                      shape: [n_entities, neighbor_sample_size]
                    - adj_relation (torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                      shape: [n_entities, neighbor_sample_size]
      :rtype: tuple



   .. py:method:: get_neighbors(items)

      Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

      :param items: The input tensor that contains item's id, shape: [batch_size, ]
      :type items: torch.LongTensor

      :returns:

                    - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                      dimensions of entities: {[batch_size, 1],
                      [batch_size, n_neighbor],
                      [batch_size, n_neighbor^2],
                      ...,
                      [batch_size, n_neighbor^n_iter]}
                    - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                      entities. Relations have the same shape as entities.
      :rtype: tuple



   .. py:method:: aggregate(user_embeddings, entities, relations)

      For each item, aggregate the entity representation and its neighborhood representation into a single vector.

      :param user_embeddings: The embeddings of users, shape: [batch_size, embedding_size]
      :type user_embeddings: torch.FloatTensor
      :param entities: entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                       dimensions of entities: {[batch_size, 1],
                       [batch_size, n_neighbor],
                       [batch_size, n_neighbor^2],
                       ...,
                       [batch_size, n_neighbor^n_iter]}
      :type entities: list
      :param relations: relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                        relations have the same shape as entities.
      :type relations: list

      :returns: The embeddings of items, shape: [batch_size, embedding_size]
      :rtype: item_embeddings(torch.FloatTensor)



   .. py:method:: label_smoothness_predict(user_embeddings, user, entities, relations)

      Predict the label of items by label smoothness.

      :param user_embeddings: The embeddings of users, shape: [batch_size*2, embedding_size],
      :type user_embeddings: torch.FloatTensor
      :param user: the index of users, shape: [batch_size*2]
      :type user: torch.FloatTensor
      :param entities: entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                       dimensions of entities: {[batch_size*2, 1],
                       [batch_size*2, n_neighbor],
                       [batch_size*2, n_neighbor^2],
                       ...,
                       [batch_size*2, n_neighbor^n_iter]}
      :type entities: list
      :param relations: relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                        relations have the same shape as entities.
      :type relations: list

      :returns: The predicted label of items, shape: [batch_size*2]
      :rtype: predicted_labels(torch.FloatTensor)



   .. py:method:: forward(user, item)


   .. py:method:: calculate_ls_loss(user, item, target)

      Calculate label smoothness loss.

      :param user: the index of users, shape: [batch_size*2],
      :type user: torch.FloatTensor
      :param item: the index of items, shape: [batch_size*2],
      :type item: torch.FloatTensor
      :param target: the label of user-item, shape: [batch_size*2],
      :type target: torch.FloatTensor

      :returns: label smoothness loss
      :rtype: ls_loss



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



.. py:class:: KGRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KGRec is a self-supervised knowledge-aware recommender that identifies and focuses on informative knowledge
   graph connections through an attentive rationalization mechanism. It combines generative masking reconstruction
   and contrastive learning tasks to highlight and align meaningful knowledge and interaction signals. By masking
   and rebuilding high-rationale edges while filtering noisy ones, KGRec learns more interpretable and noise-resistant
   recommendations.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: context_hops


   .. py:attribute:: node_dropout_rate


   .. py:attribute:: mess_dropout_rate


   .. py:attribute:: mae_coef


   .. py:attribute:: mae_msize


   .. py:attribute:: cl_coef


   .. py:attribute:: cl_tau


   .. py:attribute:: cl_drop


   .. py:attribute:: samp_func


   .. py:attribute:: inter_edge


   .. py:attribute:: kg_graph


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: mf_loss


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_entity_e
      :value: None



   .. py:attribute:: gcn


   .. py:attribute:: contrast_fn


   .. py:attribute:: node_dropout


   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data of KG.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: relation_aware_edge_sampling(sampling_rate=0.5)


   .. py:method:: edge_sampling(edge_index, edge_type, sampling_rate=0.5)


   .. py:method:: mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id)


   .. py:method:: adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score)


   .. py:method:: adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w)


   .. py:method:: create_mae_loss(node_pair_emb, masked_edge_emb=None)


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



.. py:class:: KTUP(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   KTUP is a knowledge-based recommendation model. It adopts the strategy of multi-task learning to jointly learn
   recommendation and KG-related tasks, with the goal of understanding the reasons that a user interacts with an item.
   This method utilizes an attention mechanism to combine all preferences into a single-vector representation.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: L1_flag


   .. py:attribute:: use_st_gumbel


   .. py:attribute:: kg_weight


   .. py:attribute:: align_weight


   .. py:attribute:: margin


   .. py:attribute:: user_embedding


   .. py:attribute:: item_embedding


   .. py:attribute:: pref_embedding


   .. py:attribute:: pref_norm_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: relation_norm_embedding


   .. py:attribute:: rec_loss


   .. py:attribute:: kg_loss


   .. py:attribute:: reg_loss


   .. py:method:: _masked_softmax(logits)


   .. py:method:: convert_to_one_hot(indices, num_classes)

      :param indices: A vector containing indices,
                      whose size is (batch_size,).
      :type indices: Variable
      :param num_classes: The number of classes, which would be
                          the second dimension of the resulting one-hot matrix.
      :type num_classes: Variable

      :returns: The one-hot matrix of size (batch_size, num_classes).
      :rtype: torch.Tensor



   .. py:method:: st_gumbel_softmax(logits, temperature=1.0)

      Return the result of Straight-Through Gumbel-Softmax Estimation.
      It approximates the discrete sampling via Gumbel-Softmax trick
      and applies the biased ST estimator.
      In the forward propagation, it emits the discrete one-hot result,
      and in the backward propagation it approximates the categorical
      distribution via smooth Gumbel-Softmax distribution.

      :param logits: A un-normalized probability values,
                     which has the size (batch_size, num_classes)
      :type logits: Variable
      :param temperature: A temperature parameter. The higher
                          the value is, the smoother the distribution is.
      :type temperature: float

      :returns: The sampled output, which has the property explained above.
      :rtype: torch.Tensor



   .. py:method:: _get_preferences(user_e, item_e, use_st_gumbel=False)


   .. py:method:: _transH_projection(original, norm)
      :staticmethod:



   .. py:method:: _get_score(h_e, r_e, t_e)


   .. py:method:: forward(user, item)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.

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



.. py:class:: MCCLK(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   MCCLK is a knowledge-based recommendation model.
   It focuses on the contrastive learning in KG-aware recommendation and proposes a novel multi-level cross-view
   contrastive learning mechanism. This model comprehensively considers three different graph views for KG-aware
   recommendation, including global-level structural view, local-level collaborative and semantic views. It hence
   performs contrastive learning across three views on both local and global levels, mining comprehensive graph
   feature and structure information in a self-supervised manner.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: reg_weight


   .. py:attribute:: lightgcn_layer


   .. py:attribute:: item_agg_layer


   .. py:attribute:: temperature


   .. py:attribute:: alpha


   .. py:attribute:: beta


   .. py:attribute:: loss_type


   .. py:attribute:: inter_matrix


   .. py:attribute:: inter_graph


   .. py:attribute:: kg_graph


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: gcn


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:attribute:: reg_loss


   .. py:attribute:: restore_user_e
      :value: None



   .. py:attribute:: restore_item_e
      :value: None



   .. py:method:: get_edges(graph)


   .. py:method:: forward()


   .. py:method:: light_gcn(user_embedding, item_embedding, adj)


   .. py:method:: sim(z1: torch.Tensor, z2: torch.Tensor)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: local_level_loss(A_embedding, B_embedding)


   .. py:method:: global_level_loss_1(A_embedding, B_embedding)


   .. py:method:: global_level_loss_2(A_embedding, B_embedding)


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



.. py:class:: MKR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   MKR is a Multi-task feature learning approach for Knowledge graph enhanced Recommendation. It is a deep
   end-to-end framework that utilizes knowledge graph embedding task to assist recommendation task. The two
   tasks are associated by cross&compress units, which automatically share latent features and learn high-order
   interactions between items in recommender systems and entities in the knowledge graph.


   .. py:attribute:: input_type


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: kg_embedding_size


   .. py:attribute:: L


   .. py:attribute:: H


   .. py:attribute:: reg_weight


   .. py:attribute:: use_inner_product


   .. py:attribute:: dropout_prob


   .. py:attribute:: user_embeddings_lookup


   .. py:attribute:: item_embeddings_lookup


   .. py:attribute:: entity_embeddings_lookup


   .. py:attribute:: relation_embeddings_lookup


   .. py:attribute:: user_mlp


   .. py:attribute:: tail_mlp


   .. py:attribute:: cc_unit


   .. py:attribute:: kge_mlp


   .. py:attribute:: kge_pred_mlp


   .. py:attribute:: sigmoid_BCE


   .. py:method:: forward(user_indices=None, item_indices=None, head_indices=None, relation_indices=None, tail_indices=None)


   .. py:method:: _l2_loss(inputs)


   .. py:method:: calculate_rs_loss(interaction)

      Calculate the training loss for a batch data of RS.



   .. py:method:: calculate_kg_loss(interaction)

      Calculate the training loss for a batch data of KG.



   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



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


