PLM
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3485447.3511937>`_

**Title:** Path Language Modeling over Knowledge Graphs for Explainable Recommendation

**Authors:** Shijie Geng, Zuohui Fu, Juntao Tan, Yingqiang Ge, Gerard de Melo, Yongfeng Zhang

**Abstract:** Recommender systems are ubiquitous in the online world, but their
machine-learned models often lack the ability to explain the reasons behind
the generated recommendations. To address this issue, path language modeling
over knowledge graphs has been proposed for explainable recommendation. This
approach trains language models on sequences of entity-relation triplets
extracted from knowledge graphs, enabling the generation of natural language
explanation paths. The model learns to predict the next token in a sequence
representing a path from user to item through the knowledge graph, providing
both recommendations and human-readable explanations.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : Size of the embeddings. Defaults to ``768``.
- ``num_heads (int)`` : Number of heads in the multi-head attention. Defaults to ``12``.
- ``num_layers (int)`` : Number of layers in the transformer. Defaults to ``6``.
- ``learning_rate (float)`` : The learning rate for training. Defaults to ``2e-4``.
- ``weight_decay (float)`` : Weight decay for regularization. Defaults to ``0.01``.
- ``warmup_steps (int)`` : Number of warmup steps for learning rate scheduler. Defaults to ``250``.
- ``use_kg_token_types (bool)`` : Whether to use token types for the knowledge graph. Defaults to ``True``.
- ``base_model (str)`` : The base transformer model. Defaults to ``'distilgpt2'``.
- ``sequence_postprocessor (str)`` : The postprocessor for sequence generation. Defaults to ``'Cumulative'``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='PLM', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Notes:**

- PLM requires path sampling from the knowledge graph. Ensure your dataset has KG information.
- Install the ``pathlm`` extra: ``uv pip install hopwise[pathlm]``
- PLM performs unbounded decoding, meaning it can generate paths not present in the KG.

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [1e-4,2e-4,5e-4]
   embedding_size choice [256,512,768]
   num_layers choice [4,6,8]
   num_heads choice [8,12]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

   margin choice [0.5,1.0,2.0]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

