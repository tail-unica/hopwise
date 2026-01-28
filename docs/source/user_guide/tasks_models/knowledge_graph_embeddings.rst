Knowledge Graph Embeddings
========================================

Knowledge Graph Embedding (KGE) methods learn low-dimensional vector representations
of entities and relations in a knowledge graph. These embeddings capture semantic
relationships and can be used for link prediction, recommendation, and as pre-trained
features for other models.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

KGE models learn embeddings by optimizing a scoring function that measures the
plausibility of knowledge graph triples ``(head, relation, tail)``. The goal is
to assign higher scores to true triples and lower scores to false ones.

**Use cases in Hopwise**:

1. **Link Prediction**: Predict missing links in the knowledge graph
2. **Recommendation**: Use KG structure to improve recommendations
3. **Pre-training**: Generate embeddings for path reasoning models (PGPR, CAFE, TPRec)

Model Categories
----------------

Hopwise implements several families of KGE models:

**Translation-based Models**
   These interpret relations as translations in the embedding space.

   - **TransE**: ``h + r ≈ t`` (simple translation)
   - **TransH**: Projects to relation-specific hyperplanes
   - **TransR**: Projects to relation-specific spaces
   - **TransD**: Dynamic projection matrices
   - **TorusE**: Embeddings on a torus manifold
   - **RotatE**: Relations as rotations in complex space

**Semantic Matching Models**
   These use similarity-based scoring functions.

   - **DistMult**: Diagonal bilinear model
   - **ComplEx**: Complex-valued embeddings
   - **HolE**: Circular correlation
   - **ANALOGY**: Analogical inference
   - **RESCAL**: Full bilinear model

**Neural Network Models**
   These use neural networks for scoring.

   - **ConvE**: 2D convolutional model
   - **ConvKB**: Convolutional knowledge base model

**Tensor Decomposition Models**
   - **TuckER**: Tucker decomposition


Quick Start
-----------

**Basic Training**:

.. code:: bash

   hopwise train --model=TransE --dataset=ml-100k

**With Configuration File**:

.. code:: yaml

   # transe_config.yaml
   embedding_size: 100
   margin: 1.0

   epochs: 500
   train_batch_size: 2048
   learning_rate: 0.001

   # For link prediction
   eval_args:
       split: {'RS': [0.8, 0.1, 0.1]}
   metrics: ['Hit', 'MRR']
   topk: [1, 3, 10]

.. code:: bash

   hopwise train --model=TransE --dataset=ml-100k --config_files=transe_config.yaml


Using KGE for Path Reasoning Models
-----------------------------------

One of the key uses of KGE models in Hopwise is to generate pre-trained embeddings
for Reinforcement Learning-based path reasoning models (PGPR, CAFE, TPRec).

**Workflow**:

1. **Train a KGE model** (e.g., TransE):

   .. code:: bash

      hopwise train --model=TransE --dataset=ml-100k

2. **Format the embeddings** using the provided notebook:

   See ``run_example/format_kg_aware_recsys_embs.ipynb``

   This creates ``.useremb``, ``.entityemb``, and ``.relationemb`` files.

3. **Load embeddings in path reasoning model**:

   .. code:: yaml

      # PGPR configuration
      additional_feat_suffix: [useremb, entityemb, relationemb]

      load_col:
          useremb: [user_embedding_id, user_embedding]
          entityemb: [entity_embedding_id, entity_embedding]
          relationemb: [relation_embedding_id, relation_embedding]

      preload_weight:
          user_embedding_id: user_embedding
          entity_embedding_id: entity_embedding
          relation_embedding_id: relation_embedding

For detailed instructions, see :doc:`/user_guide/explainability/rl_based_models`.


Model Selection Guide
---------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Model
     - Strengths
     - Best For
   * - **TransE**
     - Simple, fast, effective
     - General use, pre-training for PGPR/CAFE
   * - **RotatE**
     - Handles symmetric/antisymmetric relations
     - Complex relation patterns
   * - **ComplEx**
     - Handles symmetric relations well
     - Datasets with many symmetric relations
   * - **DistMult**
     - Very fast training
     - Large-scale datasets
   * - **ConvE**
     - Higher capacity
     - Complex datasets, when accuracy is priority


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   model/knowledge_graph_embeddings/transe
   model/knowledge_graph_embeddings/transh
   model/knowledge_graph_embeddings/transr
   model/knowledge_graph_embeddings/transd
   model/knowledge_graph_embeddings/toruse
   model/knowledge_graph_embeddings/rotate
   model/knowledge_graph_embeddings/tucker
   model/knowledge_graph_embeddings/distmult
   model/knowledge_graph_embeddings/analogy
   model/knowledge_graph_embeddings/hole
   model/knowledge_graph_embeddings/rescal
   model/knowledge_graph_embeddings/complex
   model/knowledge_graph_embeddings/conve
   model/knowledge_graph_embeddings/convkb


See Also
--------

- :doc:`/user_guide/explainability/rl_based_models` - Using KGE for path reasoning
- :doc:`knowledge_based_recommendation` - Knowledge-aware recommendation models