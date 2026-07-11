hopwise.model.abstract_recommender
==================================

.. py:module:: hopwise.model.abstract_recommender

.. autoapi-nested-parse::

   hopwise.model.abstract_recommender
   ##################################



Classes
-------

.. autoapisummary::

   hopwise.model.abstract_recommender.AbstractRecommender
   hopwise.model.abstract_recommender.GeneralRecommender
   hopwise.model.abstract_recommender.AutoEncoderMixin
   hopwise.model.abstract_recommender.SequentialRecommender
   hopwise.model.abstract_recommender.KnowledgeRecommender
   hopwise.model.abstract_recommender.ExplainableRecommender
   hopwise.model.abstract_recommender.PathLanguageModelingRecommender
   hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender
   hopwise.model.abstract_recommender.ContextRecommender


Module Contents
---------------

.. py:class:: AbstractRecommender(_skip_nn_module_init=False)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all models


   .. py:attribute:: logger


   .. py:method:: calculate_loss(interaction)
      :abstractmethod:


      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: predict(interaction)
      :abstractmethod:


      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict(interaction)
      :abstractmethod:


      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict_kg(interaction)
      :abstractmethod:


      Full sort prediction KG function.
      Given heads, calculate the scores between heads and all candidate tails.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given heads and all candidate tails,
                shape: [n_batch_heads * n_candidate_tails]
      :rtype: torch.Tensor



   .. py:method:: other_parameter()


   .. py:method:: load_other_parameter(para)


   .. py:method:: __str__()

      Model prints with number of trainable parameters



.. py:class:: GeneralRecommender(config, dataset)

   Bases: :py:obj:`AbstractRecommender`


   This is a abstract general recommender. All the general model should implement this class.
   The base general recommender class provide the basic dataset and parameters information.


   .. py:attribute:: type


   .. py:attribute:: USER_ID


   .. py:attribute:: ITEM_ID


   .. py:attribute:: NEG_ITEM_ID


   .. py:attribute:: n_users


   .. py:attribute:: n_items


   .. py:attribute:: device


.. py:class:: AutoEncoderMixin

   This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
   including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
   The base AutoEncoderMixin class provides basic dataset information and rating matrix function.


   .. py:method:: build_histroy_items(dataset)


   .. py:method:: get_rating_matrix(user)

      Get a batch of user's feature with the user's id and history interaction matrix.

      :param user: The input tensor that contains user's id, shape: [batch_size, ]
      :type user: torch.LongTensor

      :returns: The user's feature of a batch of user, shape: [batch_size, n_items]
      :rtype: torch.FloatTensor



.. py:class:: SequentialRecommender(config, dataset)

   Bases: :py:obj:`AbstractRecommender`


   This is a abstract sequential recommender. All the sequential model should implement This class.


   .. py:attribute:: type


   .. py:attribute:: USER_ID


   .. py:attribute:: ITEM_ID


   .. py:attribute:: ITEM_SEQ


   .. py:attribute:: ITEM_SEQ_LEN


   .. py:attribute:: POS_ITEM_ID


   .. py:attribute:: NEG_ITEM_ID


   .. py:attribute:: max_seq_length


   .. py:attribute:: n_items


   .. py:attribute:: device


   .. py:method:: gather_indexes(output, gather_index)

      Gathers the vectors at the specific positions over a minibatch



   .. py:method:: get_attention_mask(item_seq, bidirectional=False)

      Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention.



.. py:class:: KnowledgeRecommender(config, dataset, _skip_nn_module_init=False)

   Bases: :py:obj:`AbstractRecommender`


   This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
   The base knowledge-based recommender class provide the basic dataset and parameters information.


   .. py:attribute:: type


   .. py:attribute:: USER_ID


   .. py:attribute:: ITEM_ID


   .. py:attribute:: NEG_ITEM_ID


   .. py:attribute:: ENTITY_ID


   .. py:attribute:: RELATION_ID


   .. py:attribute:: HEAD_ENTITY_ID


   .. py:attribute:: TAIL_ENTITY_ID


   .. py:attribute:: NEG_TAIL_ENTITY_ID


   .. py:attribute:: n_users


   .. py:attribute:: n_items


   .. py:attribute:: n_entities


   .. py:attribute:: n_relations


.. py:class:: ExplainableRecommender

   This is a abstract explainable-based recommender. All the explainable-based model should implement this class.
   This class use templates to make the explanation more interpretable.



   .. py:method:: explain(interaction)
      :abstractmethod:


      Explain the prediction function.

      Given users, calculate the scores and paths between users and all candidate items,
      then return the templates filled with path data.

      :param interaction: The interaction batch.
      :type interaction: Interaction

      :returns:

                Predicted scores for given users and all candidate items,
                    with shape [n_batch_users * n_candidate_items].
                pandas.DataFrame: Explanation of the prediction, containing paths and corresponding templates,
                    with shape [n_paths * [uid, pid, score, template1, template2, ..., #templates]].
      :rtype: torch.Tensor



   .. py:method:: decode_path(path)
      :abstractmethod:


      Decode the path into a string. Path decoding is specific to each model.

      :param path: The path data.
      :type path: list

      :returns: The decoded path string.
      :rtype: str



.. py:class:: PathLanguageModelingRecommender(config, dataset, _skip_nn_module_init=True)

   Bases: :py:obj:`KnowledgeRecommender`


   This is an abstract path-language-modeling recommender.
   All the path-language-modeling model should implement this class.
   The base path-language-modeling recommender class inherits the knowledge-aware recommender class to
   learn from knowledge graph paths defined by a chain of entity-relation triplets.


   .. py:attribute:: type


   .. py:attribute:: input_type


   .. py:attribute:: n_tokens


   .. py:attribute:: token_sequence_length


   .. py:attribute:: logits_processor_list


   .. py:attribute:: sequence_postprocessor


   .. py:method:: generate(inputs, top_k=None, paths_per_user=1, **kwargs)

      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.

      :param inputs: A dictionary containing the input_ids tensor with shape (b, t).
      :type inputs: dict
      :param top_k: If specified, only the top k logits will be considered
                    for sampling at each step. Defaults to None.
      :type top_k: int, optional
      :param paths_per_user: How many paths to return for each user.
      :type paths_per_user: int, optional
      :param \*\*kwargs: Additional keyword arguments for the model. In future, it can be used to pass
                         other generation parameters such as temperature, repetition penalty, etc.



.. py:class:: ExplainablePathLanguageModelingRecommender(config, dataset, _skip_nn_module_init=True)

   Bases: :py:obj:`PathLanguageModelingRecommender`, :py:obj:`ExplainableRecommender`


   This is an abstract explainable path-language-modeling recommender.
   All the explainable path-language-modeling model should implement this class.
   The base explainable path-language-modeling recommender class inherits the path-language-modeling recommender class
   to learn from knowledge graph paths defined by a chain of entity-relation triplets.


   .. py:method:: explain(inputs, **kwargs)

      Explain the prediction function.

      Given users, calculate the scores and paths between users and all candidate items,
      then return the templates filled with path data.

      :param interaction: The interaction batch.
      :type interaction: Interaction

      :returns:

                Predicted scores for given users and all candidate items,
                    with shape [n_batch_users * n_candidate_items].
                pandas.DataFrame: Explanation of the prediction, containing paths and corresponding templates,
                    with shape [n_paths * [uid, pid, score, template1, template2, ..., #templates]].
      :rtype: torch.Tensor



   .. py:method:: decode_path(path)

      Standardize path format



.. py:class:: ContextRecommender(config, dataset)

   Bases: :py:obj:`AbstractRecommender`


   This is a abstract context-aware recommender. All the context-aware model should implement this class.
   The base context-aware recommender class provide the basic embedding function of feature fields which also
   contains a first-order part of feature fields.


   .. py:attribute:: type


   .. py:attribute:: input_type


   .. py:attribute:: field_names


   .. py:attribute:: LABEL


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: double_tower


   .. py:attribute:: numerical_features


   .. py:attribute:: token_field_names
      :value: []



   .. py:attribute:: token_field_dims
      :value: []



   .. py:attribute:: float_field_names
      :value: []



   .. py:attribute:: float_field_dims
      :value: []



   .. py:attribute:: token_seq_field_names
      :value: []



   .. py:attribute:: token_seq_field_dims
      :value: []



   .. py:attribute:: float_seq_field_names
      :value: []



   .. py:attribute:: float_seq_field_dims
      :value: []



   .. py:attribute:: num_feature_field
      :value: 0



   .. py:attribute:: first_order_linear


   .. py:method:: embed_float_fields(float_fields)

      Embed the float feature columns

      :param float_fields: The input dense tensor. shape of [batch_size, num_float_field]
      :type float_fields: torch.FloatTensor

      :returns: The result embedding tensor of float columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_float_seq_fields(float_seq_fields, mode='mean')

      Embed the float feature columns

      :param float_seq_fields: The input tensor. shape of [batch_size, seq_len]
      :type float_seq_fields: torch.LongTensor
      :param mode: How to aggregate the embedding of feature in this field. default=mean
      :type mode: str

      :returns: The result embedding tensor of token sequence columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_token_fields(token_fields)

      Embed the token feature columns

      :param token_fields: The input tensor. shape of [batch_size, num_token_field]
      :type token_fields: torch.LongTensor

      :returns: The result embedding tensor of token columns.
      :rtype: torch.FloatTensor



   .. py:method:: embed_token_seq_fields(token_seq_fields, mode='mean')

      Embed the token feature columns

      :param token_seq_fields: The input tensor. shape of [batch_size, seq_len]
      :type token_seq_fields: torch.LongTensor
      :param mode: How to aggregate the embedding of feature in this field. default=mean
      :type mode: str

      :returns: The result embedding tensor of token sequence columns.
      :rtype: torch.FloatTensor



   .. py:method:: double_tower_embed_input_fields(interaction)

      Embed the whole feature columns in a double tower way.

      :param interaction: The input data collection.
      :type interaction: Interaction

      :returns: The embedding tensor of token sequence columns in the first part.
                torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
                torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
                torch.FloatTensor: The embedding tensor of float sequence columns in the second part.
      :rtype: torch.FloatTensor



   .. py:method:: concat_embed_input_fields(interaction)


   .. py:method:: embed_input_fields(interaction)

      Embed the whole feature columns.

      :param interaction: The input data collection.
      :type interaction: Interaction

      :returns: The embedding tensor of token sequence columns.
                torch.FloatTensor: The embedding tensor of float sequence columns.
      :rtype: torch.FloatTensor



