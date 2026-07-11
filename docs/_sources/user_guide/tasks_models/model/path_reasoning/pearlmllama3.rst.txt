PEARLMLlama3
================

Introduction
---------------------

**Title:** Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph (Llama 3 Variant)

**Authors:** Giacomo Balloccu, Ludovico Boratto, Gianni Fenu, Mirko Marras

**Abstract:** PEARLMLlama3 is a variant of PEARLM that uses the Llama 3 architecture as the base
language model. Llama 3 is the latest generation of Meta AI's large language models, featuring
improved architecture with grouped-query attention and rotary position embeddings (RoPE).
This variant leverages Llama 3's enhanced architecture for path generation in knowledge graphs.

PEARLM (Path Explainable and Aware Reasoning Language Model) extends PLM by
adding a constrained graph decoding mechanism to ensure that generated paths are
valid according to the knowledge graph structure. The model learns entity-relation
sequences from the KG and uses attention-based masking during inference to restrict
token predictions to valid graph neighbors.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : Size of the embeddings. Defaults to ``100``.
- ``num_heads (int)`` : Number of heads in the multi-head attention. Defaults to ``1``.
- ``num_layers (int)`` : Number of layers in the transformer. Defaults to ``1``.
- ``temperature (float)`` : Temperature for sampling. Defaults to ``1.0``.
- ``rope_base (int)`` : Base frequency for RoPE (Rotary Position Embeddings). Defaults to ``10000``.
- ``rope_config`` : Configuration for RoPE. Defaults to ``None``.
- ``num_kv_groups (int)`` : Number of key-value groups for grouped-query attention. Defaults to ``8``.
- ``base_model (str)`` : The base transformer model. Defaults to ``'llama3'``.
- ``sequence_postprocessor (str)`` : The postprocessor for sequence generation. Defaults to ``'SampleSearch'``.
- ``MAX_PATHS_PER_USER (int)`` : Maximum paths per user during inference. Defaults to ``1``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='PEARLMLlama3', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Notes:**

- PEARLMLlama3 requires path sampling from the knowledge graph. Ensure your dataset has KG information.
- Install the ``pathlm`` extra: ``uv pip install hopwise[pathlm]``
- Llama 3 architecture features grouped-query attention for efficient inference.
- RoPE (Rotary Position Embeddings) enables better handling of position information.
- This variant may require more GPU memory due to the Llama 3 architecture.

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [1e-4,2e-4,5e-4]
   embedding_size choice [64,100,200]
   num_layers choice [1,2,3]
   num_heads choice [1,2,4]
   num_kv_groups choice [4,8,16]
   temperature choice [0.5,1.0,1.5]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.
