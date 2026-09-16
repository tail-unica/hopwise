hopwise.model.path_language_modeling_recommender.plm
====================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.plm

.. autoapi-nested-parse::

   PLM
   ################################################
   Reference:
        Shijie Geng et al. "Path Language Modeling over Knowledge Graphs for Explainable Recommendation." in WWW 2022.

   Reference code:
       https://github.com/mirkomarras/kgglm



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.plm.PLM


Module Contents
---------------

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



