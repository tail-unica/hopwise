PEARLM
===========

Introduction
---------------------

`[paper] <https://github.com/Chris1nexus/pearlm>`_

**Title:** Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph

**Authors:** Giacomo Balloccu, Ludovico Boratto, Gianni Fenu, Mirko Marras

**Abstract:** PEARLM (Path Explainable and Aware Reasoning Language Model) extends PLM by
adding a constrained graph decoding mechanism to ensure that generated paths are
valid according to the knowledge graph structure. Unlike PLM which performs
unbounded decoding, PEARLM constrains the generation process to only produce
paths that actually exist in the knowledge graph, making the explanations more
faithful and interpretable. The model learns entity-relation sequences from
the KG and uses attention-based masking during inference to restrict token
predictions to valid graph neighbors.


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
- ``MAX_PATHS_PER_USER (int)`` : Maximum paths per user during inference. Defaults to ``1``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='PEARLM', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Notes:**

- PEARLM requires path sampling from the knowledge graph. Ensure your dataset has KG information.
- Install the ``pathlm`` extra: ``uv pip install hopwise[pathlm]``
- PEARLM ensures graph-faithful decoding, generating only valid KG paths.

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

