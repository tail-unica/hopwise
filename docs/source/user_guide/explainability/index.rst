Explainability
==============

Hopwise is specifically designed for **explainable recommendation systems**. This section covers the specialized models and workflows for generating human-interpretable explanations alongside recommendations.

Hopwise supports two main categories of explainable models:

1. **Reinforcement Learning-based Path Reasoning Models** (PGPR, CAFE, TPRec) - These models learn to traverse knowledge graphs to find explanation paths connecting users to recommended items.

2. **Language Model-based Explainers** (PLM, PEARLM, KGGLM) - These models generate natural language explanations or leverage language model architectures for path generation.

.. toctree::
   :maxdepth: 2

   rl_based_models
   lm_based_models


Why Explainability Matters
--------------------------

Traditional recommendation systems are often "black boxes" - they suggest items but don't explain *why*. Explainable recommendations provide:

- **Trust**: Users understand and trust recommendations they can verify
- **Transparency**: The reasoning process is visible and auditable
- **Debugging**: Developers can identify and fix model issues
- **Compliance**: Meets regulatory requirements (e.g., GDPR's right to explanation)

Knowledge Graph-Based Explanations
----------------------------------

Hopwise leverages knowledge graphs (KGs) to generate explanations. A knowledge graph connects entities (users, items, attributes) through semantic relationships:

.. code:: text

   User_123 --[watched]--> Movie_A --[directed_by]--> Director_X --[directed]--> Movie_B

This path explains why User_123 might like Movie_B: "Because you watched Movie_A, which was directed by Director_X, who also directed Movie_B."

Getting Started
---------------

To use explainability models in Hopwise:

1. **Prepare your dataset** with knowledge graph files (``.kg`` and ``.link`` files)
2. **Choose your model** based on your needs (see the subsections below)
3. **Train and evaluate** using the Hopwise CLI or Python API

See the individual model sections for detailed setup instructions.
