Introduction to Hopwise
===================

Welcome to **hopwise**, an advanced extension of the RecBole library designed to enhance
recommendation systems with the power of **knowledge graphs**.

What is Hopwise?
----------------

Hopwise integrates knowledge embedding models, path-based reasoning methods, and path language
modeling approaches to support both recommendation and link prediction tasks with a focus on
**explainability**.

.. image:: ../asset/hopwise.png
    :width: 600
    :align: center

Key Features
------------

Hopwise extends the popular RecBole framework with several unique capabilities:

1. **Path Reasoning Methods**: Novel recommendation models that leverage knowledge graph paths
   to provide explainable recommendations (KGGLM, PEARLM, PLM).

2. **Knowledge Graph Embeddings**: 14 KGE methods for both recommendation and link prediction
   tasks (TransE, TransH, TransR, TransD, RotatE, ComplEx, DistMult, etc.).

3. **Path Quality Metrics**: New evaluation metrics for assessing the quality of explanation
   paths (LIR, SEP, SED, PTD, PTC, etc.).

4. **Path Sampling Utilities**: Tools for sampling and managing paths from knowledge graphs.

5. **Link Prediction Support**: Full support for knowledge graph completion tasks alongside
   recommendation.

Model Categories
----------------

Hopwise supports six categories of recommendation models:

- **General Recommendation**: Collaborative filtering models (BPR, NeuMF, LightGCN, etc.)
- **Sequential Recommendation**: Models that consider interaction sequences (SASRec, BERT4Rec, etc.)
- **Context-aware Recommendation**: Models that use contextual features (DeepFM, xDeepFM, etc.)
- **Knowledge-based Recommendation**: Models that leverage knowledge graphs (KGAT, KGIN, RippleNet, etc.)
- **Path Reasoning Recommendation**: Explainable models using KG paths (KGGLM, PEARLM, PLM)
- **Knowledge Graph Embeddings**: KGE models for recommendation and link prediction

Quick Example
-------------

Here's a simple example to train a knowledge-aware model:

.. code:: python

    from hopwise.quick_start import run_hopwise

    # Train a knowledge-aware model
    run_hopwise(model='KGAT', dataset='ml-100k')

For path-based models with language modeling:

.. code:: python

    from hopwise.quick_start import run_hopwise

    # Train a path language model
    run_hopwise(model='KGGLM', dataset='ml-100k')

Command Line Interface
----------------------

Hopwise provides a convenient CLI for common tasks:

.. code:: bash

    # Train a model
    hopwise train --model KGAT --dataset ml-100k

    # Evaluate from checkpoint
    hopwise evaluate --model KGAT --dataset ml-100k --checkpoint saved/model.pth

    # Hyperparameter tuning
    hopwise tune --model KGAT --dataset ml-100k --params_file hyper.test

Next Steps
----------

- :doc:`../get_started/install`: Installation instructions
- :doc:`../user_guide/usage/run_hopwise`: Running your first model
- :doc:`../user_guide/tasks_models/path_reasoning_recommendation`: Learn about path reasoning models
- :doc:`../user_guide/tasks_models/knowledge_graph_embeddings`: Explore KGE models