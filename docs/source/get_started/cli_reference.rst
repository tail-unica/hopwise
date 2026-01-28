Command Line Interface
======================

Hopwise provides a powerful command-line interface (CLI) for training, evaluating, and tuning recommendation models.
This section covers all available commands and their options.

Installation
------------

When you install Hopwise, the CLI is automatically available:

.. code:: bash

   pip install hopwise
   # or with uv
   uv pip install hopwise

After installation, verify it works:

.. code:: bash

   hopwise --version
   hopwise --help


Global Options
--------------

These options are available for all commands:

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--debug``
     - Enable debug mode with full tracebacks
     - ``False``
   * - ``--rich-traceback``
     - Use Rich's enhanced traceback formatting with local variables
     - ``False``
   * - ``--version``
     - Show the installed version
     -
   * - ``--help``
     - Show help message
     -

Example with debug mode:

.. code:: bash

   hopwise --debug train --model BPR --dataset ml-100k


Commands
--------

hopwise train
~~~~~~~~~~~~~

Train a model on a dataset.

.. code:: bash

   hopwise train [OPTIONS]

**Options:**

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--model, -m``
     - Model name to train (e.g., BPR, LightGCN, KGAT)
     - ``BPR``
   * - ``--dataset, -d``
     - Dataset name
     - ``ml-100k``
   * - ``--config-files``
     - Space-separated configuration files
     - ``None``
   * - ``--checkpoint``
     - Path to checkpoint file (.pth) for resuming training
     - ``None``
   * - ``--nproc``
     - Number of processes for distributed training
     - ``1``
   * - ``--ip``
     - Master node IP address for distributed training
     - ``localhost``
   * - ``--port``
     - Master node port for distributed training
     - ``5678``
   * - ``--world-size``
     - Total number of distributed jobs
     - ``-1``
   * - ``--group-offset``
     - Global rank offset
     - ``0``
   * - ``--proc-title``
     - Process title shown in system monitors (top, nvidia-smi)
     - Auto-generated

**Examples:**

Basic training:

.. code:: bash

   hopwise train --model BPR --dataset ml-100k

Training with configuration files:

.. code:: bash

   hopwise train --model KGAT --dataset ml-100k --config-files "config/kgat.yaml config/dataset.yaml"

Resume training from checkpoint:

.. code:: bash

   hopwise train --model BPR --dataset ml-100k --checkpoint saved/BPR-ml-100k.pth

Distributed training:

.. code:: bash

   hopwise train --model LightGCN --dataset ml-1m --nproc 4

You can also pass Hopwise configuration parameters directly:

.. code:: bash

   hopwise train --model BPR --dataset ml-100k --epochs=50 --learning_rate=0.001


hopwise evaluate
~~~~~~~~~~~~~~~~

Evaluate a pre-trained model from a checkpoint.

.. code:: bash

   hopwise evaluate [OPTIONS]

**Options:**

The options are the same as ``hopwise train``. The key difference is that ``--checkpoint`` is typically required to load a pre-trained model.

**Example:**

.. code:: bash

   hopwise evaluate --model KGAT --dataset ml-100k --checkpoint saved/KGAT-ml-100k.pth


hopwise benchmark
~~~~~~~~~~~~~~~~~

Run experiments across multiple models and generate LaTeX tables for scientific publications.

.. code:: bash

   hopwise benchmark [OPTIONS]

**Options:**

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--models, -m``
     - Comma-separated list of model names (required)
     -
   * - ``--dataset, -d``
     - Dataset name
     - ``ml-100k``
   * - ``--config-files``
     - Space-separated configuration files
     - ``None``
   * - ``--valid-latex``
     - Output path for validation results LaTeX file
     - ``./latex/valid.tex``
   * - ``--test-latex``
     - Output path for test results LaTeX file
     - ``./latex/test.tex``
   * - ``--nproc``
     - Number of processes
     - ``1``

**Example:**

.. code:: bash

   hopwise benchmark --models "BPR,LightGCN,KGAT,PGPR" --dataset ml-100k

This will:

1. Train each model sequentially on the dataset
2. Collect validation and test results
3. Generate LaTeX tables comparing all models

The generated LaTeX tables are ready to include in scientific papers.


hopwise tune
~~~~~~~~~~~~

Run hyperparameter tuning using Optuna, Hyperopt, or Ray Tune.

.. code:: bash

   hopwise tune PARAMS-FILE [OPTIONS]

**Arguments:**

- ``PARAMS-FILE``: Path to a YAML file defining the hyperparameter search space

**Options:**

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--config-files``
     - Fixed configuration files (non-tunable parameters)
     - ``None``
   * - ``--output-path``
     - Directory to save tuning results
     - ``saved/hyper``
   * - ``--display-file``
     - Path for visualization file
     - ``None``
   * - ``--max-evals``
     - Maximum number of evaluations
     - ``10``
   * - ``--tool``
     - Tuning framework: ``hyperopt``, ``ray``, or ``optuna``
     - ``optuna``
   * - ``--study-name``
     - Name for the tuning study
     - Auto-generated with timestamp
   * - ``--algo``
     - Search algorithm (tool-specific)
     - Tool default
   * - ``--resume``
     - Resume from a previous checkpoint
     - ``False``

**Hyperparameter file format (hyper.test example):**

.. code:: yaml

   learning_rate loguniform -8 0
   embedding_size choice [32, 64, 128, 256]
   train_batch_size choice [256, 512, 1024, 2048]

**Example:**

.. code:: bash

   hopwise tune hyper.test --tool optuna --max-evals 50 --study-name my_experiment

Using Ray Tune for distributed tuning:

.. code:: bash

   hopwise tune hyper.test --tool ray --max-evals 100 --output-path saved/ray_results


hopwise models
~~~~~~~~~~~~~~

List all available models in Hopwise.

.. code:: bash

   hopwise models [OPTIONS]

**Options:**

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--verbose``
     - Show detailed model descriptions (docstrings)
     - ``False``
   * - ``--type``
     - Filter by model type (can be used multiple times)
     - ``all``

**Available model types:**

- ``all`` - Show all models (default)
- ``General`` - General recommendation models (BPR, LightGCN, etc.)
- ``Sequential`` - Sequential recommendation models (SASRec, BERT4Rec, etc.)
- ``Context`` - Context-aware models (DeepFM, DCN, etc.)
- ``KG-aware`` - Knowledge graph-aware models (KGAT, KGCN, etc.)
- ``KG-embed`` - Knowledge graph embedding models (TransE, RotatE, etc.)
- ``PathLM`` - Path language modeling models (PLM, PEARLM, KGGLM)
- ``Exlib`` - Explainability library models

**Examples:**

List all models:

.. code:: bash

   hopwise models

List only knowledge graph models with descriptions:

.. code:: bash

   hopwise models --type KG-aware --type KG-embed --verbose

List path-based models:

.. code:: bash

   hopwise models --type PathLM


Quick Start Examples
--------------------

Here are some common workflows:

**1. Train your first model:**

.. code:: bash

   hopwise train --model BPR --dataset ml-100k

**2. Train a knowledge-aware model:**

.. code:: bash

   hopwise train --model KGAT --dataset ml-100k --config-files "config/kgat.yaml"

**3. Train and compare multiple models:**

.. code:: bash

   hopwise benchmark --models "BPR,LightGCN,NGCF" --dataset ml-100k

**4. Tune hyperparameters:**

.. code:: bash

   hopwise tune hyper.test --tool optuna --max-evals 30

**5. Evaluate a saved model:**

.. code:: bash

   hopwise evaluate --model BPR --dataset ml-100k --checkpoint saved/BPR-ml-100k.pth


Configuration Override
----------------------

Hopwise CLI supports passing configuration parameters directly on the command line using the ``--key=value`` syntax:

.. code:: bash

   hopwise train --model BPR --dataset ml-100k --epochs=100 --learning_rate=0.001 --embedding_size=64

This is equivalent to setting these values in a YAML configuration file:

.. code:: yaml

   epochs: 100
   learning_rate: 0.001
   embedding_size: 64

The priority order (highest to lowest):

1. Command-line parameters (``--key=value``)
2. Configuration files (``--config-files``)
3. Model default configuration
4. Dataset default configuration
5. Global default configuration


Exit Codes
----------

The CLI uses standard exit codes:

- ``0``: Success
- ``1``: General error
- ``130``: Interrupted by user (Ctrl+C)


See Also
--------

- :doc:`/user_guide/configuration` - Detailed configuration guide
- :doc:`/user_guide/hyperparameters_tuning` - Advanced hyperparameter tuning
- :doc:`/user_guide/training_and_evaluation` - Training strategies and evaluation
