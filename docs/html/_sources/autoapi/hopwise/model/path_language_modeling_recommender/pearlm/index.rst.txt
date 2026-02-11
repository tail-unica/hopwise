hopwise.model.path_language_modeling_recommender.pearlm
=======================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.pearlm

.. autoapi-nested-parse::

   PEARLM
   ##################################################
   Reference:
       Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

   Reference code:
       https://github.com/Chris1nexus/pearlm



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlm.PEARLM


Module Contents
---------------

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



