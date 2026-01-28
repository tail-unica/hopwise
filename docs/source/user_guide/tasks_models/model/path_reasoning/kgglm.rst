KGGLM
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3640457.3688138>`_

**Title:** KGGLM: A Generative Language Model for Generalizable Knowledge Graph Representation Learning in Recommendation

**Authors:** Giacomo Balloccu, Ludovico Boratto, Gianni Fenu, Mirko Marras

**Abstract:** Knowledge graph-enhanced recommendations have gained popularity due
to their ability to leverage structured information for better user/item representations.
This paper introduces KGGLM, a generative language model that learns generalizable
knowledge graph representations for recommendation. By pre-training on knowledge
graph paths and fine-tuning on user-item interactions, KGGLM can generate
reasoning paths that explain recommendations.

KGGLM follows a two-stage training process:

1. **Pre-training**: Learn to generate valid knowledge graph paths
2. **Fine-tuning**: Adapt the model for user-item recommendation

Running with Hopwise
-------------------------

.. note::

   The language model is currently hardcoded to use **DistilGPT-2**. While a ``base_model``
   parameter exists, changing it is not supported as the tokenizer and model architecture
   are tied to DistilGPT-2. This may change in future versions.

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : Size of the embeddings. Defaults to ``768``.
- ``num_heads (int)`` : Number of attention heads. Defaults to ``12``.
- ``num_layers (int)`` : Number of transformer layers. Defaults to ``6``.
- ``train_stage (str)`` : Training stage. Range in ``['pretrain', 'finetune']``.
- ``pre_model_path (str)`` : Path to pre-trained model for fine-tuning.
- ``warmup_steps (int)`` : Number of warmup steps. Defaults to ``250``.
- ``pretrain_epochs (int)`` : Number of pre-training epochs. Defaults to ``1``.
- ``save_step (int)`` : Steps between checkpoints. Defaults to ``50``.


**Step 1: Pre-training**

Create ``kgglm_pretrain.yaml``:

.. code:: yaml

   # KGGLM Pre-training Configuration
   train_stage: 'pretrain'
   embedding_size: 768
   num_heads: 12
   num_layers: 6

   learning_rate: 2e-4
   weight_decay: 0.01
   warmup_steps: 250
   pretrain_epochs: 1
   save_step: 50

   path_sample_args:
       strategy: simple-ui
       pretrain_hop_length: [3, 3]
       pretrain_paths: 1

Run pre-training:

.. code:: bash

   hopwise train --model=KGGLM --dataset=ml-100k --config_files=kgglm_pretrain.yaml

**Step 2: Fine-tuning**

Create ``kgglm_finetune.yaml``:

.. code:: yaml

   # KGGLM Fine-tuning Configuration
   train_stage: 'finetune'
   pre_model_path: 'saved/KGGLM-pretrain-xxx/'

   learning_rate: 1e-4
   epochs: 10

Run fine-tuning:

.. code:: bash

   hopwise train --model=KGGLM --dataset=ml-100k --config_files=kgglm_finetune.yaml

**Generating Explanations:**

.. code:: python

   from hopwise.quick_start import load_data_and_model

   config, model, dataset, _, _, test_data = load_data_and_model(
       model_file='saved/KGGLM-xxx.pth'
   )

   # Generate explanation paths
   user_id = 1
   explanations = model.explain(user_id)
   for exp in explanations:
       print(exp)

Tuning Hyper Parameters
-------------------------

.. code:: bash

   learning_rate choice [1e-4,2e-4,5e-4]
   num_layers choice [4,6,8]
   num_heads choice [8,12]
   warmup_steps choice [100,250,500]

Then run:

.. code:: bash

   hopwise tune --model=KGGLM --dataset=ml-100k --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

See Also
--------

- :doc:`/user_guide/explainability/lm_based_models` - Complete LM models guide
- :doc:`pearlm` - PEARLM model (parent class)
- :doc:`plm` - PLM model (simpler approach)

