PLM
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3485447.3511937>`_

**Title:** Path Language Modeling over Knowledge Graphs for Explainable Recommendation

**Authors:** Shijie Geng, Zuohui Fu, Juntao Tan, Yingqiang Ge, Gerard de Melo, Yongfeng Zhang

**Abstract:** Providing explanations for recommendations has received increasing attention
in recent years. Knowledge graphs (KGs) can provide rich information for generating
explanations, as the paths in a KG connecting a user and an item can reveal the reasons
behind a recommendation. This paper introduces PLM (Path Language Model), which treats
the paths as sequences and learns to generate them using a language model.

PLM learns to predict sequences of entity-relation triplets from the knowledge graph.
Note that PLM's decoding is unbounded, meaning it can generate paths that don't exist
in the original knowledge graph.

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


**Configuration Example:**

.. code:: yaml

   # PLM Configuration
   embedding_size: 768
   num_heads: 12
   num_layers: 6
   use_kg_token_types: True

   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250

   path_sample_args:
       restrict_by_phase: False

**Running Example:**

.. code:: bash

   hopwise train --model=PLM --dataset=ml-100k --config_files=plm_config.yaml

Or via Python:

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='PLM', dataset='ml-100k')

**Generating Explanations:**

.. code:: python

   from hopwise.quick_start import load_data_and_model

   config, model, dataset, _, _, test_data = load_data_and_model(
       model_file='saved/PLM-xxx.pth'
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

   hopwise tune --model=PLM --dataset=ml-100k --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

See Also
--------

- :doc:`/user_guide/explainability/lm_based_models` - Complete LM models guide
- :doc:`pearlm` - PEARLM model (adds constrained decoding)
- :doc:`kgglm` - KGGLM model (extends PEARLM with two-stage training)

