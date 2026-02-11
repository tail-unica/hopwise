hopwise.model.path_language_modeling_recommender.pearlmgpt2
===========================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.pearlmgpt2

.. autoapi-nested-parse::

   PEARLMGPT2
   ##################################################
   Reference:
       Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph." - preprint.

   Reference code:
       https://github.com/Chris1nexus/pearlm
       https://github.com/karpathy/nanoGPT/blob/master/model.py
       https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.pearlmgpt2.LayerNorm
   hopwise.model.path_language_modeling_recommender.pearlmgpt2.AutoregressiveSelfAttention
   hopwise.model.path_language_modeling_recommender.pearlmgpt2.FeedForward
   hopwise.model.path_language_modeling_recommender.pearlmgpt2.Block
   hopwise.model.path_language_modeling_recommender.pearlmgpt2.PEARLMGPT2


Module Contents
---------------

.. py:class:: LayerNorm(ndim, bias)

   Bases: :py:obj:`torch.nn.Module`


   LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False


   .. py:attribute:: weight


   .. py:attribute:: bias


   .. py:method:: forward(input)


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


   .. py:attribute:: attn_dropout


   .. py:attribute:: resid_dropout


   .. py:attribute:: causal_mask


   .. py:method:: forward(x)


.. py:class:: FeedForward(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: c_fc


   .. py:attribute:: silu


   .. py:attribute:: c_proj


   .. py:attribute:: dropout


   .. py:method:: forward(x)


.. py:class:: Block(config)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: layernorm_1


   .. py:attribute:: causal_attn


   .. py:attribute:: layernorm_2


   .. py:attribute:: feedforward


   .. py:method:: forward(x)


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



