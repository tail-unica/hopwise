Path Reasoning Recommendation
=======================================

Path reasoning methods leverage knowledge graph structure to create explainable
recommendation paths from users to items. These methods provide not just recommendations
but also human-interpretable explanations of *why* an item is recommended.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

Path reasoning models traverse the knowledge graph to find meaningful paths
connecting users to items. For example:

.. code::

   User → purchased → Movie A → directed_by → Director X → directed → Movie B (recommended)

This path explains that "Movie B is recommended because you liked Movie A, and
both are directed by Director X."


Model Categories
----------------

Hopwise supports two main families of path reasoning models:

**Reinforcement Learning (RL) Models**
   Train an agent to navigate the knowledge graph and find optimal paths.

   - **PGPR**: Policy-Guided Path Reasoning
   - **CAFE**: Coarse-to-fine path reasoning
   - **TPRec**: Temporal Path Reasoning

   *Detailed documentation*: :doc:`/user_guide/explainability/rl_based_models`

**Language Model (LM) Models**
   Use language models to generate path-based explanations.

   - **PLM**: Path Language Model
   - **PEARLM**: Path Extraction and Reasoning with Language Models
   - **KGGLM**: Knowledge Graph-Guided Language Model

   *Detailed documentation*: :doc:`/user_guide/explainability/lm_based_models`


Quick Start
-----------

**RL-based Model (PGPR)**:

PGPR and other RL models require pre-trained KG embeddings from TransE or similar models.

.. code:: bash

   # Step 1: Train KGE model
   hopwise train --model=TransE --dataset=ml-100k

   # Step 2: Format embeddings (see format_kg_aware_recsys_embs.ipynb)

   # Step 3: Train PGPR
   hopwise train --model=PGPR --dataset=ml-100k

**LM-based Model (KGGLM)**:

.. code:: bash

   hopwise train --model=KGGLM --dataset=ml-100k --config_files=kgglm_config.yaml

Where ``kgglm_config.yaml`` contains:

.. code:: yaml

   # KGGLM configuration
   train_stage: pretrain  # or 'generate'
   path_sample_args:
       pretrain_hop_length: 3
       pretrain_paths: 100

See :doc:`/user_guide/explainability/lm_based_models` for detailed configuration.


Comparison of Approaches
------------------------

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - Aspect
     - RL-based (PGPR, CAFE)
     - LM-based (PLM, KGGLM)
   * - **Mechanism**
     - Agent learns navigation policy
     - Language model generates paths
   * - **Explainability**
     - Path sequences
     - Natural language descriptions
   * - **Pre-requisites**
     - KGE embeddings required
     - No external embeddings needed
   * - **Training**
     - Policy gradient methods
     - Standard LM fine-tuning
   * - **Scalability**
     - Efficient for large graphs
     - Limited by LM context length


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   model/path_reasoning/kgglm
   model/path_reasoning/pearlm
   model/path_reasoning/plm


See Also
--------

- :doc:`/user_guide/explainability/index` - Detailed explainability documentation
- :doc:`/user_guide/explainability/rl_based_models` - RL-based path reasoning
- :doc:`/user_guide/explainability/lm_based_models` - LM-based path reasoning
- :doc:`knowledge_graph_embeddings` - KGE for pre-training
- :doc:`/user_guide/tasks_models/model/knowledge_aware/pgpr` - PGPR model details
- :doc:`/user_guide/tasks_models/model/knowledge_aware/cafe` - CAFE model details