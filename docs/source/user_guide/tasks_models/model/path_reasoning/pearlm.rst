PEARLM
===========

Introduction
---------------------

`[paper] <https://arxiv.org/abs/2403.16032>`_

**Title:** Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph

**Authors:** Giacomo Balloccu, Ludovico Boratto, Christian Cancedda, Gianni Fenu, Mirko Marras

**Abstract:** PEARLM (Path-language-modeling Explainable Recommendation over Language Model)
extends the PLM approach by adding a constrained graph decoding mechanism to ensure
that the generated paths are valid according to the knowledge graph structure. This
results in more faithful and reliable explanation paths for recommendations.

PEARLM learns to predict sequences of entity-relation triplets extracted from a
knowledge graph, and can generate valid reasoning paths that explain recommendations.

Running with Hopwise
-------------------------

.. note::

   The language model is currently hardcoded to use **DistilGPT-2**. While a ``base_model``
   parameter exists, changing it is not supported as the tokenizer and model architecture
   are tied to DistilGPT-2. This may change in future versions.

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : Size of the embeddings. Defaults to ``768``.
- ``num_heads (int)`` : Number of attention heads. Defaults to ``12``.
- ``num_layers (int)`` : Number of transformer layers. Defaults to ``6``.
- ``use_kg_token_types (bool)`` : Whether to use token types for KG. Defaults to ``True``.
- ``warmup_steps (int)`` : Number of warmup steps. Defaults to ``250``.
- ``MAX_PATHS_PER_USER (int)`` : Maximum paths per user. Defaults to ``1``.


**Configuration Example:**

.. code:: yaml

   # PEARLM Configuration
   embedding_size: 768
   num_heads: 12
   num_layers: 6
   use_kg_token_types: True

   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250

   MAX_PATHS_PER_USER: 1
   path_sample_args:
       temporal_causality: False
       strategy: simple-ui

**Running Example:**

.. code:: bash

   hopwise train --model=PEARLM --dataset=ml-100k --config_files=pearlm_config.yaml

**Generating Explanations:**

.. code:: python

   from hopwise.quick_start import load_data_and_model

   config, model, dataset, _, _, test_data = load_data_and_model(
       model_file='saved/PEARLM-xxx.pth'
   )

   # Generate explanation paths
   user_id = 1
   explanations = model.generate(user_id)
   for exp in explanations:
       print(exp)

Tuning Hyper Parameters
-------------------------

.. code:: bash

   learning_rate choice [1e-4,2e-4,5e-4]
   num_layers choice [4,6,8]
   num_heads choice [8,12]
   embedding_size choice [256,512,768]

Then run:

.. code:: bash

   hopwise tune --model=PEARLM --dataset=ml-100k --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

See Also
--------

- :doc:`/user_guide/explainability/lm_based_models` - Complete LM models guide
- :doc:`kgglm` - KGGLM model (extends PEARLM with two-stage training)
- :doc:`plm` - PLM model (simpler approach without constrained decoding)

