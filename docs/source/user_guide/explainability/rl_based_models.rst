Reinforcement Learning-based Models
====================================

This section covers the **Reinforcement Learning (RL)-based path reasoning models** in Hopwise: **PGPR**, **CAFE**, and **TPRec**. These models learn to traverse knowledge graphs using policy gradient methods to generate explanation paths for recommendations.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

RL-based models treat recommendation as a sequential decision-making problem where an agent navigates the knowledge graph:

1. **State**: Current position in the knowledge graph
2. **Action**: Select the next relation and entity to traverse
3. **Reward**: Based on whether the path leads to a relevant item

These models require **pre-trained knowledge graph embeddings** to initialize their entity and relation representations. This is a crucial step that significantly impacts model performance.


Supported Models
----------------

.. list-table::
   :widths: 15 50 35
   :header-rows: 1

   * - Model
     - Description
     - Paper
   * - **PGPR**
     - Policy-Guided Path Reasoning. Uses policy gradients to learn path selection.
     - `Xian et al., KDD 2019 <https://arxiv.org/abs/1906.05237>`_
   * - **CAFE**
     - Coarse-to-Fine Path Reasoning. Hierarchical approach with entity abstraction.
     - `Xian et al., KDD 2020 <https://arxiv.org/abs/2007.02173>`_
   * - **TPRec**
     - Temporal Path Reasoning. Incorporates temporal dynamics into path reasoning.
     - `Fu et al., WSDM 2024 <https://arxiv.org/abs/2312.06976>`_


Prerequisites: Pre-trained KG Embeddings
----------------------------------------

RL-based models require pre-trained embeddings from a Knowledge Graph Embedding (KGE) model. Hopwise provides 14 KGE models including TransE, RotatE, ComplEx, and more.

Step 1: Train a KGE Model
~~~~~~~~~~~~~~~~~~~~~~~~~

First, train a KGE model on your dataset to generate embeddings:

.. code:: bash

   hopwise train --model TransE --dataset ml-100k --config-files "config/kge.yaml"

Example ``config/kge.yaml``:

.. code:: yaml

   # KGE training configuration
   embedding_size: 64
   epochs: 500
   learning_rate: 0.001
   train_neg_sample_args:
     distribution: uniform
     sample_num: 1
     alpha: 1.0
     dynamic: False

After training, a checkpoint file will be saved (e.g., ``saved/TransE-ml-100k-xxx.pth``).

Step 2: Format Embeddings for Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the provided notebook or script to convert the KGE checkpoint into the embedding format that Hopwise can load:

.. code:: python

   import torch
   import pandas as pd
   import os
   from hopwise.quick_start import create_dataset

   # Load the trained KGE checkpoint
   checkpoint_name = "saved/TransE-ml-100k-xxx.pth"
   checkpoint = torch.load(checkpoint_name)

   # View available embeddings
   print(checkpoint['state_dict'].keys())
   # Typically: ['entity_embedding.weight', 'relation_embedding.weight']

   # Get dataset info
   config = checkpoint['config']
   dataset = create_dataset(config)
   dataset_name = config['dataset']
   data_path = config['data_path']

   # Create reverse mappings (internal ID -> original ID)
   eid2token = {id: token for token, id in dataset.field2token_id['entity_id'].items()}
   rid2token = {id: token for token, id in dataset.field2token_id['relation_id'].items()}

   def format_embedding(weight, id2token, emb_type, dataset_name, data_path):
       """Convert embedding tensor to Hopwise format."""
       weight = weight.detach().cpu().numpy()

       # Skip padding token (index 0)
       ids = [id2token[i] for i in range(1, weight.shape[0])]
       embeddings = [" ".join(f"{x}" for x in row) for row in weight[1:]]

       df = pd.DataFrame({
           f'{emb_type}_embedding_id:token': ids,
           f'{emb_type}_embedding:float_seq': embeddings
       })

       filename = f'{dataset_name}.{emb_type}emb'
       filepath = os.path.join(data_path, filename)
       df.to_csv(filepath, sep='\t', index=False)
       print(f"[+] Saved {emb_type} embeddings to {filepath}")

   # Format and save entity embeddings
   entity_emb = checkpoint['state_dict']['entity_embedding.weight']
   format_embedding(entity_emb, eid2token, 'entity', dataset_name, data_path)

   # Format and save relation embeddings
   relation_emb = checkpoint['state_dict']['relation_embedding.weight']
   format_embedding(relation_emb, rid2token, 'relation', dataset_name, data_path)

After running this script, you'll have:

- ``dataset/ml-100k/ml-100k.entityemb`` - Entity embeddings file
- ``dataset/ml-100k/ml-100k.relationemb`` - Relation embeddings file

.. note::

   A complete Jupyter notebook for this process is available at ``run_example/format_kg_aware_recsys_embs.ipynb``.

Step 3: Configure Pre-trained Embedding Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update your configuration file to load the pre-trained embeddings:

.. code:: yaml

   # config/pgpr_pretrained.yaml

   # Specify additional feature files to load
   additional_feat_suffix: [entityemb, relationemb]

   # Define which columns to load from each file
   load_col:
     inter: [user_id, item_id, timestamp]
     kg: [head_id, relation_id, tail_id]
     link: [item_id, entity_id]
     entityemb: [entity_embedding_id, entity_embedding]
     relationemb: [relation_embedding_id, relation_embedding]

   # Map embedding IDs to internal fields
   alias_of_entity_id: [entity_embedding_id]
   alias_of_relation_id: [relation_embedding_id]

   # Configure preload weights
   preload_weight:
     entity_embedding_id: entity_embedding
     relation_embedding_id: relation_embedding


Training RL-based Models
------------------------

Once you have pre-trained embeddings configured, train the RL model:

PGPR
~~~~

.. code:: bash

   hopwise train --model PGPR --dataset ml-100k --config-files "config/pgpr_pretrained.yaml"

PGPR-specific configuration options:

.. code:: yaml

   # PGPR configuration
   embedding_size: 64
   hidden_size: [512, 256]
   learning_rate: 0.0001
   max_path_length: 3
   sample_size: 40
   state_history: 1
   gamma: 0.99
   ent_weight: 0.01
   act_dropout: 0.5

CAFE
~~~~

.. code:: bash

   hopwise train --model CAFE --dataset ml-100k --config-files "config/cafe_pretrained.yaml"

CAFE-specific configuration options:

.. code:: yaml

   # CAFE configuration
   embedding_size: 64
   hidden_sizes: [512, 256]
   learning_rate: 0.0001
   max_path_length: 3
   gamma: 0.99
   ent_weight: 0.01
   sample_size: 50
   num_neg_samples: 5

TPRec
~~~~~

.. code:: bash

   hopwise train --model TPRec --dataset ml-100k --config-files "config/tprec_pretrained.yaml"

TPRec incorporates temporal information, so ensure your dataset includes timestamps.


Complete End-to-End Example
---------------------------

Here's a complete workflow from raw data to trained RL model:

.. code:: bash

   # 1. Prepare dataset (ensure you have .inter, .kg, .link files)
   # See Data Module documentation for format details

   # 2. Train KGE model for embeddings
   hopwise train --model TransE --dataset my_dataset --epochs=500

   # 3. Format embeddings (run the Python script above)
   # This creates .entityemb and .relationemb files

   # 4. Train RL model with pre-trained embeddings
   hopwise train --model PGPR --dataset my_dataset --config-files "config/pgpr_pretrained.yaml"

   # 5. Evaluate
   hopwise evaluate --model PGPR --dataset my_dataset --checkpoint saved/PGPR-my_dataset-xxx.pth


Extracting Explanation Paths
----------------------------

After training, you can extract explanation paths for recommendations:

.. code:: python

   from hopwise.quick_start import load_model_and_dataset

   # Load trained model
   model, dataset, config = load_model_and_dataset("saved/PGPR-ml-100k-xxx.pth")

   # Get paths for a specific user
   user_id = 42
   paths, scores = model.get_explanation_paths(user_id, top_k=10)

   # Each path is a sequence: [user] -> [relation] -> [entity] -> ... -> [item]
   for path, score in zip(paths, scores):
       print(f"Score: {score:.4f}")
       print(" -> ".join(str(node) for node in path))


Tips and Best Practices
-----------------------

1. **KGE Model Selection**: TransE works well for simple datasets; try RotatE or ComplEx for datasets with complex relation patterns.

2. **Embedding Dimension**: Use the same ``embedding_size`` for both KGE pre-training and RL training.

3. **Path Length**: Start with ``max_path_length=3`` and increase if your KG is sparse.

4. **Training Stability**: RL training can be unstable. Use a low learning rate (1e-4 to 1e-5) and consider gradient clipping.

5. **Negative Sampling**: More negative samples generally improve performance but increase training time.


Troubleshooting
---------------

**Assertion Error during embedding loading**

If you get an assertion error about embedding dimensions not matching:

.. code:: text

   AssertionError: Embedding size mismatch

Ensure you trained the KGE model **without** adding dummy relations/entities explicitly.

**No paths found for user**

If the model finds no paths:

- Check that the user has interactions in the training set
- Verify the KG is properly connected (items should link to entities)
- Try increasing ``sample_size`` or ``max_path_length``


See Also
--------

- :doc:`/user_guide/tasks_models/knowledge_based_recommendation` - Overview of knowledge-aware models
- :doc:`/user_guide/tasks_models/knowledge_graph_embeddings` - KGE models documentation
- :doc:`/user_guide/usage/load_pretrained_embedding` - General pre-trained embedding loading
