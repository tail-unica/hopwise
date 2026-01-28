Introduction to Hopwise
=======================

Welcome to Hopwise! This tutorial will guide you through the basics of using Hopwise
for building knowledge-enhanced recommendation systems.

What is Hopwise?
----------------

Hopwise is an advanced extension of the RecBole library, designed to enhance
recommendation systems with the power of knowledge graphs. It supports:

- **Traditional Recommendation**: General, Sequential, Context-aware models
- **Knowledge-enhanced Recommendation**: Knowledge Graph Embedding models
- **Explainable Recommendation**: Path Reasoning and Path Language Modeling

Quick Start Example
-------------------

Let's start with a simple example to train a recommendation model:

.. code:: python

    from hopwise.quick_start import run_hopwise

    # Train a simple BPR model on ml-100k dataset
    result = run_hopwise(model='BPR', dataset='ml-100k')
    print(result)

Or use the CLI:

.. code:: bash

    hopwise train --model=BPR --dataset=ml-100k

Understanding the Output
------------------------

When you run a model, Hopwise will:

1. **Load and preprocess the data** - Convert raw data to internal format
2. **Initialize the model** - Create model with specified hyperparameters
3. **Train the model** - Optimize on training set
4. **Validate** - Evaluate on validation set for early stopping
5. **Test** - Report final metrics on test set

Example output:

.. code::

    INFO  BPR(
      (user_embedding): Embedding(944, 64)
      (item_embedding): Embedding(1683, 64)
    )
    INFO  epoch 0 training [time: 0.21s, train loss: 27.7228]
    INFO  epoch 0 evaluating [time: 0.92s, valid_score: 0.020500]
    ...
    INFO  test result: {'recall@10': 0.2523, 'mrr@10': 0.4855}

Configuration System
--------------------

Hopwise uses YAML configuration files. Create a ``config.yaml``:

.. code:: yaml

    # Dataset settings
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    load_col:
        inter: [user_id, item_id]

    # Model settings
    embedding_size: 64

    # Training settings
    epochs: 100
    train_batch_size: 2048
    learning_rate: 0.001

    # Evaluation settings
    eval_args:
        split: {'RS': [0.8, 0.1, 0.1]}
        mode: full
    metrics: ['Recall', 'NDCG', 'MRR', 'Hit']
    topk: 10
    valid_metric: MRR@10

Then run:

.. code:: bash

    hopwise train --model=BPR --dataset=ml-100k --config_files=config.yaml

Working with Knowledge Graphs
-----------------------------

Hopwise excels at knowledge-enhanced recommendation. Here's how to use
knowledge graph information:

**1. Knowledge Graph Embedding Models (e.g., TransE)**

These models learn embeddings for entities and relations:

.. code:: python

    run_hopwise(model='TransE', dataset='ml-100k')

**2. Knowledge-aware Recommendation (e.g., CKE)**

These combine KG embeddings with recommendation:

.. code:: python

    run_hopwise(model='CKE', dataset='ml-100k')

**3. Explainable Models (e.g., PGPR)**

Path-based reasoning models require pre-trained embeddings.
See :doc:`/user_guide/explainability/rl_based_models` for details.

Next Steps
----------

- :doc:`/get_started/cli_reference` - Learn the CLI commands
- :doc:`/user_guide/configuration` - Deep dive into configuration
- :doc:`/user_guide/data_intro` - Understand data formats
- :doc:`/user_guide/explainability/index` - Explore explainable models