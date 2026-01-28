PGPR
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3331184.3331203>`_

**Title:** Reinforcement Knowledge Graph Reasoning for Explainable Recommendation

**Authors:** Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang

**Abstract:** Recent advances in knowledge graph-based recommendations have
demonstrated the potential of incorporating structured knowledge into recommender
systems. However, most existing methods focus on utilizing knowledge graph
embeddings for better user/item representations, without exploring the reasoning
capabilities that knowledge graphs could provide. In this paper, we propose
PGPR (Policy-Guided Path Reasoning), a reinforcement learning-based model that
reasons over knowledge graph paths for explainable recommendation.

PGPR uses an actor-critic architecture to learn a policy that navigates
the knowledge graph to find reasoning paths from users to items.

.. image:: ../../../asset/pgpr.png
    :width: 600
    :align: center

Running with Hopwise
-------------------------

**Prerequisites:**

PGPR requires pre-trained knowledge graph embeddings. You must first train
a KGE model (e.g., TransE) and format the embeddings. See
:doc:`/user_guide/explainability/rl_based_models` for the complete workflow.

**Model Hyper-Parameters:**

- ``state_history (int)`` : Number of historical states to track. Defaults to ``1``.
- ``max_acts (int)`` : Maximum number of actions at each step. Defaults to ``250``.
- ``gamma (float)`` : Discount factor for rewards. Defaults to ``0.99``.
- ``action_dropout (float)`` : Dropout rate for action selection. Defaults to ``0.0``.
- ``hidden_sizes (list)`` : Hidden layer sizes for the policy network. Defaults to ``[512, 256]``.
- ``max_path_len (int)`` : Maximum path length. Defaults to ``3``.
- ``weight_factor (float)`` : Weight for entropy loss. Defaults to ``1e-3``.
- ``beam_search_hop (list)`` : Beam width at each hop for inference. Defaults to ``[25, 5, 1]``.


**Step 1: Train KGE Model**

.. code:: bash

   hopwise train --model=TransE --dataset=ml-100k --config_files=transe_config.yaml

**Step 2: Format Embeddings**

Use the notebook at ``run_example/format_kg_aware_recsys_embs.ipynb`` to format
embeddings into ``.useremb``, ``.entityemb``, and ``.relationemb`` files.

**Step 3: Train PGPR**

Create ``pgpr_config.yaml``:

.. code:: yaml

   # PGPR Configuration
   state_history: 1
   max_acts: 250
   gamma: 0.99
   hidden_sizes: [512, 256]
   max_path_len: 3
   beam_search_hop: [25, 5, 1]

   # Pre-trained embedding files
   additional_feat_suffix: [useremb, entityemb, relationemb]

   load_col:
       useremb: [user_embedding_id, user_embedding]
       entityemb: [entity_embedding_id, entity_embedding]
       relationemb: [relation_embedding_id, relation_embedding]

   preload_weight:
       user_embedding_id: user_embedding
       entity_embedding_id: entity_embedding
       relation_embedding_id: relation_embedding

Run training:

.. code:: bash

   hopwise train --model=PGPR --dataset=ml-100k --config_files=pgpr_config.yaml

**Extracting Explanation Paths:**

.. code:: python

   from hopwise.quick_start import load_data_and_model

   config, model, dataset, _, _, test_data = load_data_and_model(
       model_file='saved/PGPR-xxx.pth'
   )

   # Get paths for a user
   user_id = 1
   paths = model.get_paths(user_id, topk=10)
   for path in paths:
       print(path)

Tuning Hyper Parameters
-------------------------

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   max_acts choice [100,250,500]
   hidden_sizes choice [[256,128],[512,256]]
   gamma choice [0.9,0.95,0.99]

Then run:

.. code:: bash

   hopwise tune --model=PGPR --dataset=ml-100k --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

See Also
--------

- :doc:`/user_guide/explainability/rl_based_models` - Complete workflow guide
- :doc:`cafe` - CAFE model (similar approach)
- :doc:`tprec` - TPRec model (similar approach)

