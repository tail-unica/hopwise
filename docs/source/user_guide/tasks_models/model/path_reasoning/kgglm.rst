KGGLM
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3640457.3688119>`_

**Title:** KGGLM: A Generative Language Model for Generalizable Knowledge Graph Representation Learning in Recommendation

**Authors:** Giacomo Balloccu, Ludovico Boratto, Gianni Fenu, Mirko Marras

**Abstract:** KGGLM (Knowledge Graph Generative Language Model) introduces a two-stage training
procedure for path-based explainable recommendation. In the pre-training stage, the
model learns general knowledge graph representations by training on paths sampled
from the entire knowledge graph. In the fine-tuning stage, the model is adapted to
the specific recommendation task using user-item interaction paths. This approach
allows the model to generalize better across different datasets and domains while
providing explainable recommendations through generated knowledge graph paths.


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
- ``train_stage (str)`` : The training stage. Defaults to ``'pretrain'``. Range in ``['pretrain', 'finetune']``.
- ``pre_model_path (str)`` : The path of pretrained model (required for finetune stage). Defaults to ``''``.
- ``pretrain_epochs (int)`` : Number of epochs for pre-training. Defaults to ``1``.
- ``save_step (int)`` : Number of steps to save the model during pre-training. Defaults to ``50``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

**Pre-training:**

.. code:: python

   from hopwise.quick_start import run_hopwise

   config_dict = {
       'train_stage': 'pretrain',
       'pretrain_epochs': 5,
   }
   run_hopwise(model='KGGLM', dataset='ml-100k', config_dict=config_dict)

**Fine-tuning:**

.. code:: python

   from hopwise.quick_start import run_hopwise

   config_dict = {
       'train_stage': 'finetune',
       'pre_model_path': 'saved/KGGLM-pretrain/',
   }
   run_hopwise(model='KGGLM', dataset='ml-100k', config_dict=config_dict)

And then:

.. code:: bash

   python run.py

**Notes:**

- KGGLM uses a two-stage training process: pre-training on KG paths, then fine-tuning for recommendation.
- Install the ``pathlm`` extra: ``uv pip install hopwise[pathlm]``
- For best results, pre-train on a large portion of the knowledge graph first.

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [1e-4,2e-4,5e-4]
   embedding_size choice [256,512,768]
   num_layers choice [4,6,8]
   num_heads choice [8,12]
   pretrain_epochs choice [1,3,5]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

