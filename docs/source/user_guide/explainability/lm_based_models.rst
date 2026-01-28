Language Model-based Explainers
================================

This section covers the **Language Model (LM)-based explainability models** in Hopwise.
These models leverage language modeling techniques to generate explanation paths for recommendations.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

LM-based models treat path generation as a sequence modeling problem, where knowledge graph
paths are represented as token sequences. This approach offers several advantages:

- **Flexible path lengths**: LM models naturally handle variable-length sequences
- **Efficient inference**: Autoregressive generation is typically faster than tree search
- **Natural language generation**: Models can produce human-readable explanations

Hopwise implements three LM-based models:

1. **PLM**: Path Language Model - basic path generation without graph constraints
2. **PEARLM**: Adds constrained decoding to ensure valid graph paths
3. **KGGLM**: Extends PEARLM with two-stage pre-training and fine-tuning

.. note::

   These models are currently **hardcoded to use DistilGPT-2** as the base architecture.
   While the ``base_model`` parameter exists in configuration, switching to other models
   may cause issues as the implementation modifies internal weights and token embeddings
   specifically for DistilGPT-2. Future versions may support additional base models.


Model Architecture
------------------

All three models share a common architecture:

- **Base**: DistilGPT-2 (6 layers, 12 heads, 768 embedding size by default)
- **Vocabulary**: Extended with KG entities and relations as tokens
- **Token types**: Special tokens for entities, relations, and control tokens (BOS, EOS)
- **Training objective**: Next-token prediction on KG paths

The key differences:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Model
     - Decoding
     - Training
   * - **PLM**
     - Unconstrained (may generate invalid paths)
     - Single-stage training
   * - **PEARLM**
     - Constrained to valid KG paths
     - Single-stage training
   * - **KGGLM**
     - Constrained to valid KG paths
     - Two-stage: pretrain + finetune


PLM (Path Language Model)
-------------------------

PLM learns to predict sequences of entity-relation triplets from the knowledge graph.
Its decoding is **unconstrained**, meaning it may generate paths that don't exist in the KG.

**Reference**: Geng et al. "Path Language Modeling over Knowledge Graphs for Explainable Recommendation" (WWW 2022)

**Configuration** (``PLM.yaml``):

.. code:: yaml

   # Model architecture (hardcoded to distilgpt2)
   base_model: distilgpt2
   embedding_size: 768
   num_heads: 12
   num_layers: 6
   use_kg_token_types: True

   # Training
   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250

   # Sequence postprocessing
   sequence_postprocessor: Cumulative

   # Path sampling (nested dictionary)
   path_sample_args:
       restrict_by_phase: False

**Training**:

.. code:: bash

   hopwise train --model=PLM --dataset=ml-100k


PEARLM
------

PEARLM extends PLM with **constrained graph decoding** to ensure generated paths are valid
according to the knowledge graph structure.

**Reference**: Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph" (preprint)

**Configuration** (``PEARLM.yaml``):

.. code:: yaml

   # Model architecture (hardcoded to distilgpt2)
   base_model: distilgpt2
   embedding_size: 768
   num_heads: 12
   num_layers: 6
   use_kg_token_types: True

   # Training
   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250

   # Sequence postprocessing
   sequence_postprocessor: Cumulative

   # Path limits
   MAX_PATHS_PER_USER: 1

   # Path sampling (nested dictionary)
   path_sample_args:
       temporal_causality: False
       strategy: simple-ui

**Training**:

.. code:: bash

   hopwise train --model=PEARLM --dataset=ml-100k


KGGLM
-----

KGGLM combines knowledge graph structure with language model generation using a
**two-stage training process**: pre-training on KG paths, then fine-tuning for recommendation.

**Reference**: Balloccu et al. "KGGLM: A Generative Language Model for Generalizable Knowledge Graph Representation Learning in Recommendation" (RecSys 2024)

**Configuration** (``KGGLM.yaml``):

.. code:: yaml

   # Model architecture (hardcoded to distilgpt2)
   base_model: distilgpt2
   embedding_size: 768
   num_heads: 12
   num_layers: 6
   use_kg_token_types: True

   # Training
   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250

   # Two-stage training
   train_stage: 'pretrain'        # 'pretrain' or 'finetune'
   pre_model_path: ''             # Path to pretrained model (for finetune stage)
   pretrain_epochs: 1
   save_step: 50

   # Sequence postprocessing
   sequence_postprocessor: Cumulative

   # Path sampling (nested dictionary - note the indentation!)
   path_sample_args:
       restrict_by_phase: False
       pretrain_hop_length: [3, 3]   # Min and max hop length
       pretrain_paths: 1             # Paths per entity
       temporal_causality: False
       strategy: simple-ui
       MAX_RW_TRIES_PER_IID: 1

Training Workflow
~~~~~~~~~~~~~~~~~

**Step 1: Pre-training**

.. code:: yaml

   # kgglm_pretrain.yaml
   train_stage: 'pretrain'
   pretrain_epochs: 1
   save_step: 50

   path_sample_args:
       pretrain_hop_length: [3, 3]
       pretrain_paths: 1
       strategy: simple-ui

.. code:: bash

   hopwise train --model=KGGLM --dataset=ml-100k --config_files=kgglm_pretrain.yaml

**Step 2: Fine-tuning**

.. code:: yaml

   # kgglm_finetune.yaml
   train_stage: 'finetune'
   pre_model_path: 'saved/KGGLM-pretrain-xxx/'

.. code:: bash

   hopwise train --model=KGGLM --dataset=ml-100k --config_files=kgglm_finetune.yaml


Configuration Notes
-------------------

.. warning::

   Parameters inside ``path_sample_args`` must be **indented** as shown above.
   They are read as a nested dictionary. Placing them at the top level will cause errors.

**Correct**:

.. code:: yaml

   path_sample_args:
       strategy: simple-ui
       temporal_causality: False

**Incorrect**:

.. code:: yaml

   # These will NOT be read correctly!
   strategy: simple-ui
   temporal_causality: False


Memory Considerations
---------------------

LM-based models can be memory-intensive due to the transformer architecture:

- **Default DistilGPT-2**: ~82M parameters
- **With extended vocabulary**: Additional memory for entity/relation embeddings

**Tips for reducing memory**:

.. code:: yaml

   # Smaller batch size with gradient accumulation
   train_batch_size: 8

   # Reduce embedding size (requires retraining)
   embedding_size: 256
   num_layers: 4
   num_heads: 4


See Also
--------

- :doc:`rl_based_models` - RL-based path reasoning (PGPR, CAFE, TPRec)
- :doc:`/user_guide/tasks_models/path_reasoning_recommendation` - Overview of all path reasoning models
- :doc:`/user_guide/half_precision_training` - Using mixed precision for large models
