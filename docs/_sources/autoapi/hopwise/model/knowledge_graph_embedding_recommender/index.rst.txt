hopwise.model.knowledge_graph_embedding_recommender
===================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/analogy/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/complex/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/conve/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/convkb/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/distmult/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/hole/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/rescal/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/rotate/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/toruse/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/transd/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/transe/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/transh/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/transr/index
   /autoapi/hopwise/model/knowledge_graph_embedding_recommender/tucker/index


Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.Analogy
   hopwise.model.knowledge_graph_embedding_recommender.ComplEx
   hopwise.model.knowledge_graph_embedding_recommender.ConvE
   hopwise.model.knowledge_graph_embedding_recommender.ConvKB
   hopwise.model.knowledge_graph_embedding_recommender.DistMult
   hopwise.model.knowledge_graph_embedding_recommender.HolE
   hopwise.model.knowledge_graph_embedding_recommender.RESCAL
   hopwise.model.knowledge_graph_embedding_recommender.RotatE
   hopwise.model.knowledge_graph_embedding_recommender.TorusE
   hopwise.model.knowledge_graph_embedding_recommender.TransD
   hopwise.model.knowledge_graph_embedding_recommender.TransE
   hopwise.model.knowledge_graph_embedding_recommender.TransH
   hopwise.model.knowledge_graph_embedding_recommender.TransR
   hopwise.model.knowledge_graph_embedding_recommender.TuckER


Package Contents
----------------

.. py:class:: Analogy(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   Analogy extends RESCAL so as to further model the analogical properties of entities and relations e.g.
   Interstellar is to Fantasy as Nolan is to Oppenheimer”.
   It employs the same scoring function as RESCAL but with some constraints.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: scalar_share


   .. py:attribute:: ui_relation


   .. py:attribute:: scalar_dim


   .. py:attribute:: complex_dim


   .. py:attribute:: user_embedding


   .. py:attribute:: user_re_embedding


   .. py:attribute:: user_im_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: entity_re_embedding


   .. py:attribute:: entity_im_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: relation_re_embedding


   .. py:attribute:: relation_im_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head_e, head_re_e, head_im_e, r_e, r_re_e, r_im_e, tail_e, tail_re_e, tail_im_e)


   .. py:method:: _get_rec_embeddings(users, pos_items, neg_items)


   .. py:method:: _get_kg_embeddings(heads, relations, pos_tails, neg_tails)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: ComplEx(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   ComplEx extends DistMult by introducing complex-valued embeddings.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_re_embedding


   .. py:attribute:: user_im_embedding


   .. py:attribute:: entity_re_embedding


   .. py:attribute:: entity_im_embedding


   .. py:attribute:: relation_re_embedding


   .. py:attribute:: relation_im_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head_re_e, head_im_e, rec_r_re_e, rec_r_im_e, tail_re_e, tail_im_e)


   .. py:method:: triple_dot(x, y, z)


   .. py:method:: _get_rec_embeddings(user, positive_items, negative_items)


   .. py:method:: _get_kg_embeddings(head, relation, positive_tails, negative_tails)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: ConvE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   ConvE represent h,r,t in a subset of real number in d dimension. When scoring them,
   it concatenates and reshape h and r into a unique input [h;r]. This input is passed through
   a convolutional layers with a set of k filters and then through a dense layer with d neurons
   and a set of weight W. The output is finally combined with the tail embedding t
   using the dot product to produce the final score.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: label_smoothing


   .. py:attribute:: input_dropout


   .. py:attribute:: hidden_dropout


   .. py:attribute:: feature_dropout


   .. py:attribute:: embedding_dim1


   .. py:attribute:: embedding_dim2


   .. py:attribute:: hidden_size


   .. py:attribute:: use_bias


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relations_embeddings


   .. py:attribute:: inp_drop


   .. py:attribute:: hidden_drop


   .. py:attribute:: feature_map_drop


   .. py:attribute:: conv1


   .. py:attribute:: bn0


   .. py:attribute:: bn1


   .. py:attribute:: bn2


   .. py:attribute:: fc


   .. py:attribute:: loss


   .. py:method:: forward(head, relation, embeddings, bias)


   .. py:method:: _get_rec_embeddings(user)


   .. py:method:: _get_kg_embeddings(head, relation)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: ConvKB(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   ConvKB: The main differences from ConvE are that when scoring h, r and t,
   it concatenates them into a d x 3 matrix. This output undergoes convolution
   by a set of omega of T filters of shape 1x3, resulting in a Tx3 feature map.
   This feature map goes through a dense layer with one neuron and weights W, resulting in the final score.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: out_channels


   .. py:attribute:: kernel_size


   .. py:attribute:: drop_prob


   .. py:attribute:: lmbda


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: conv1_bn


   .. py:attribute:: conv_layer


   .. py:attribute:: conv2_bn


   .. py:attribute:: dropout


   .. py:attribute:: non_linearity


   .. py:attribute:: fc_layer


   .. py:attribute:: loss


   .. py:attribute:: reg


   .. py:method:: forward(head, relation, tail)


   .. py:method:: _get_regularization(head, relation, tail)


   .. py:method:: _get_rec_embeddings(user, positive_items, negative_items)


   .. py:method:: _get_kg_embeddings(head, relation, positive_tails, negative_tails)


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



   .. py:method:: predict_kg(interaction)


.. py:class:: DistMult(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   DistMult simplify RESCAL by restricting Mr to diagonal matrices.
   For each relation r, it introduce a vector embedding r and requires Mr = diag(r).

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head, relation, tail)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: HolE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   HoLE combines the expressive power of RESCAL with the efficiency and simplicity of DistMult.
   The entity representations are composed into h ⋆ t in the set of real numbers,
   with the circular correlation operator.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: sigmoid


   .. py:attribute:: loss


   .. py:method:: forward(h, r, t)


   .. py:method:: get_rolling_matrix(x)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: RESCAL(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   RESCAL associates each entity with a vector to capture its latent semantics.
   Each relation is represented as a matrix which models pairwise interactions between latent vectors

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head, relation, tail)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: RotatE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   RotatE models relations as rotations in a complex latent space with h, r, t belonging
   to the set of d-dimensional complex numbers. The embedding for r belonging to the set of d-dimensional
   complex numbers, is a rotation vector: in all its elements, the phase conveys the rotation along that axis,
   and the modulus is equal to 1.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: user_embedding_im


   .. py:attribute:: entity_embedding


   .. py:attribute:: entity_embedding_im


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: forward(head_re, head_im, relation, tail_re, tail_im)


   .. py:method:: _get_rec_embeddings(user, positive_items, negative_items)


   .. py:method:: _get_kg_embeddings(head, relation, positive_tails, negative_tails)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: TorusE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TorusE projects each point in a Torus.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: forward(head, relation, tail)


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: predict_kg(interaction)


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



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: TransD(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransD simplifies TransR by further decomposing the projection matrix into a product of two vector.
   Also in this case, the scoring function is the same as TransH and TransR,
   but it introduces three additional mapping vectors along with the entity and relation representation.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: user_vec_embedding


   .. py:attribute:: entity_vec_embedding


   .. py:attribute:: relation_vec_embedding


   .. py:attribute:: loss


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_rec_vec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: _get_kg_vec_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: forward(ent, ent_vect, rel_vect)

      We note that :math:`p_r(e)_i = e^p^Te \times r^p_i + e_i` which is
      more efficient to compute than the matrix formulation in the original
      paper.



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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: TransE(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransE a method which models relationships by interpreting them
   as translations operating on the low-dimensional embeddings of the entities.
   Originally created for the knowledge completion task, was adapted to make recommendation

   .. math::
       f_t(r)=(h+r,t)

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: loss


   .. py:method:: forward(user, relation, item)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict_kg(interaction)

      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



.. py:class:: TransH(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransH Have been invented to overcome the disadvantages of TransE,
   allowing an entity to have distinct representations when involved in different relations.
   It introduces relation-specific hyperplanes.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: norm_vec


   .. py:attribute:: rec_loss


   .. py:method:: forward(head, relation, tail, relation_ids)


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, pos_tail, neg_tail, relation)


   .. py:method:: project(ent, rel)


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



.. py:class:: TransR(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TransR Rather than introducing relation-specific hyperplanes, it introduces relation-specific spaces.
   The scoring functions is the same as TransH but h and t are projected into the space specific to relation

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: margin


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: proj_mat_e


   .. py:attribute:: loss


   .. py:method:: _get_rec_embedding(user, pos_item, neg_item)


   .. py:method:: _get_kg_embedding(head, relation, pos_tail, neg_tail)


   .. py:method:: forward(ent, proj_mat)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



.. py:class:: TuckER(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   TuckER relies on Tucker Decomposition. It handles entity and relation embeddings of independent dimension
   and jointly learns a share core W.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: ui_relation


   .. py:attribute:: label_smoothing


   .. py:attribute:: input_dropout


   .. py:attribute:: input_dropout1


   .. py:attribute:: input_dropout2


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: weights


   .. py:attribute:: hidden_dropout1


   .. py:attribute:: hidden_dropout2


   .. py:attribute:: bn0


   .. py:attribute:: bn1


   .. py:attribute:: loss


   .. py:method:: forward(h, r, embeddings)


   .. py:method:: _get_rec_embeddings(user)


   .. py:method:: _get_kg_embeddings(h, r)


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



   .. py:method:: predict_kg(interaction)


   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



