hopwise.model.path_language_modeling_recommender
================================================

.. py:module:: hopwise.model.path_language_modeling_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/path_language_modeling_recommender/kgglm/index
   /autoapi/hopwise/model/path_language_modeling_recommender/pearlm/index
   /autoapi/hopwise/model/path_language_modeling_recommender/pearlmgpt2/index
   /autoapi/hopwise/model/path_language_modeling_recommender/pearlmllama2/index
   /autoapi/hopwise/model/path_language_modeling_recommender/pearlmllama3/index
   /autoapi/hopwise/model/path_language_modeling_recommender/plm/index


Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.KGGLM
   hopwise.model.path_language_modeling_recommender.PEARLM
   hopwise.model.path_language_modeling_recommender.PEARLMGPT2
   hopwise.model.path_language_modeling_recommender.PEARLMLlama2
   hopwise.model.path_language_modeling_recommender.PEARLMLlama3
   hopwise.model.path_language_modeling_recommender.PLM


Package Contents
----------------

.. py:class:: KGGLM(config, dataset)

   Bases: :py:obj:`hopwise.model.path_language_modeling_recommender.pearlm.PEARLM`


   PEARLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
   as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
   representing a path. The model extends PLM by adding a constrained graph decoding mechanism to ensure that
   the generated paths are valid according to the knowledge graph structure. The model can be used for
   explainable recommendation by generating paths that explain the recommendations made by the model.


   .. py:attribute:: TRAIN_STAGES
      :value: ['pretrain', 'finetune']



   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


.. py:class:: PEARLM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender`, :py:obj:`transformers.GPT2LMHeadModel`


   PEARLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
   as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
   representing a path. The model extends PLM by adding a constrained graph decoding mechanism to ensure that
   the generated paths are valid according to the knowledge graph structure. The model can be used for
   explainable recommendation by generating paths that explain the recommendations made by the model.


   .. py:attribute:: use_kg_token_types


   .. py:attribute:: loss


   .. py:method:: forward(input_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None, attention_mask: Optional[torch.FloatTensor] = None, token_type_ids: Optional[torch.LongTensor] = None, position_ids: Optional[torch.LongTensor] = None, head_mask: Optional[torch.FloatTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs) -> Union[tuple, transformers.modeling_outputs.CausalLMOutputWithCrossAttentions]


   .. py:method:: predict(input_ids, **kwargs)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: generate(inputs, **kwargs)

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



.. py:class:: PEARLMGPT2(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender`


   Low-level implementation of PEARLM model based on GPT-2 architecture that does not rely on HuggingFace tools.


   .. py:attribute:: temperature


   .. py:attribute:: type_emb_pos


   .. py:attribute:: wte


   .. py:attribute:: wpe


   .. py:attribute:: wp_type_e


   .. py:attribute:: blocks


   .. py:attribute:: layernorm


   .. py:attribute:: dropout


   .. py:attribute:: lm_head


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(idx)


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



.. py:class:: PEARLMLlama2(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender`


   Low-level implementation of PEARLM model based on Llama2 architecture.

   Novelties:
   - LayerNorm is replaced with RMSNorm
   - GeLU is replaced with SiLU
   - Feedforward is replaced with a simple linear head


   .. py:attribute:: temperature


   .. py:attribute:: type_emb_pos


   .. py:attribute:: wte


   .. py:attribute:: wpe


   .. py:attribute:: blocks


   .. py:attribute:: rmsnorm


   .. py:attribute:: lm_head


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(idx)


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



.. py:class:: PEARLMLlama3(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender`


   Low-level implementation of PEARLM model based on LLaMA 3 architecture.

   With 8 kv-groups (that's how many Llama 3 8B uses), we can see that the number of rows
   of the key and value matrices are reduced by a factor of 4
   (because 32 attention heads divided by 8 kv-groups is 4)
   To make the GroupedQueryAttention equivalent to standard multi-head attention,
   you can set the number of query groups equal to the number of heads.


   .. py:attribute:: temperature


   .. py:attribute:: weight_precision


   .. py:attribute:: type_emb_pos


   .. py:attribute:: wte


   .. py:attribute:: wpe


   .. py:attribute:: blocks


   .. py:attribute:: rmsnorm


   .. py:attribute:: lm_head


   .. py:attribute:: loss


   .. py:method:: _init_weights(module)


   .. py:method:: forward(idx)


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



.. py:class:: PLM(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.ExplainablePathLanguageModelingRecommender`, :py:obj:`transformers.GPT2LMHeadModel`


   PLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
   from a knowledge graph as a next-token prediction task and employs two feature transformations separately
   for entities and relations. Its decoding process is unbounded, meaning that it can generate paths that are
   not faithful to the knowledge graph structure, i.e., it can generate paths that do not exist in the KG.


   .. py:attribute:: use_kg_token_types


   .. py:attribute:: n_tokens


   .. py:attribute:: token_type_ids


   .. py:attribute:: token_entity_type_id


   .. py:attribute:: token_relation_type_id


   .. py:attribute:: entity_head


   .. py:attribute:: relation_head


   .. py:attribute:: entity_loss


   .. py:attribute:: relation_loss


   .. py:method:: forward(input_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None, attention_mask: Optional[torch.FloatTensor] = None, token_type_ids: Optional[torch.LongTensor] = None, position_ids: Optional[torch.LongTensor] = None, head_mask: Optional[torch.FloatTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs) -> Union[tuple, transformers.modeling_outputs.CausalLMOutputWithCrossAttentions]


   .. py:method:: predict(input_ids, **kwargs)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: generate(inputs, **kwargs)

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



