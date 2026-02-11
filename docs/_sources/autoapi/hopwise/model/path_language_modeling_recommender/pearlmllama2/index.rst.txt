hopwise.model.path_language_modeling_recommender.pearlmllama2
=============================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.pearlmllama2

.. autoapi-nested-parse::

   PEARLMLlama2
   ##################################################
   Reference:
       Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

   Reference code:
       https://github.com/Chris1nexus/pearlm
       https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlmllama2.AutoregressiveSelfAttention
   hopwise.model.path_language_modeling_recommender.pearlmllama2.FeedForward
   hopwise.model.path_language_modeling_recommender.pearlmllama2.Block
   hopwise.model.path_language_modeling_recommender.pearlmllama2.PEARLMLlama2


Functions
---------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlmllama2.precompute_rope_params
   hopwise.model.path_language_modeling_recommender.pearlmllama2.compute_rope


Module Contents
---------------

.. py:class:: AutoregressiveSelfAttention(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: hidden_size


   .. py:attribute:: num_heads


   .. py:attribute:: dropout


   .. py:attribute:: head_dim


   .. py:attribute:: W_query


   .. py:attribute:: W_key


   .. py:attribute:: W_value


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



.. py:function:: precompute_rope_params(head_dim, theta_base=10000, context_length=4096, device=None)

.. py:function:: compute_rope(x, cos, sin)

