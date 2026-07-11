hopwise.model.path_language_modeling_recommender.pearlmllama3
=============================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.pearlmllama3

.. autoapi-nested-parse::

   PEARLMLlama3
   ##################################################
   Reference:
       Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

   Reference code:
       https://github.com/Chris1nexus/pearlm
       https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlmllama3.AutoregressiveGroupQuerySelfAttention
   hopwise.model.path_language_modeling_recommender.pearlmllama3.FeedForward
   hopwise.model.path_language_modeling_recommender.pearlmllama3.Block
   hopwise.model.path_language_modeling_recommender.pearlmllama3.PEARLMLlama3
   hopwise.model.path_language_modeling_recommender.pearlmllama3.SharedBuffers


Functions
---------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlmllama3.precompute_rope_params
   hopwise.model.path_language_modeling_recommender.pearlmllama3.compute_rope


Module Contents
---------------

.. py:class:: AutoregressiveGroupQuerySelfAttention(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: hidden_size


   .. py:attribute:: num_heads


   .. py:attribute:: dropout


   .. py:attribute:: head_dim


   .. py:attribute:: W_key


   .. py:attribute:: W_value


   .. py:attribute:: group_size


   .. py:attribute:: W_query


   .. py:attribute:: out_proj


   .. py:attribute:: causal_mask


   .. py:method:: forward(x)


.. py:class:: FeedForward(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:attribute:: silu


   .. py:method:: forward(x)


.. py:class:: Block(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: rmsnorm1


   .. py:attribute:: causal_attn


   .. py:attribute:: rmsnorm2


   .. py:attribute:: feedforward


   .. py:method:: forward(x)


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



.. py:function:: precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None, device=None)

.. py:function:: compute_rope(x, cos, sin)

.. py:class:: SharedBuffers

   .. py:attribute:: _buffers


   .. py:method:: get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32)
      :staticmethod:



