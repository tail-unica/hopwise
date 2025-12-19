ConvE
===========

Introduction
---------------------

`[paper] <https://ojs.aaai.org/index.php/AAAI/article/view/11573>`_

**Title:** Convolutional 2D Knowledge Graph Embeddings

**Authors:** Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel

**Abstract:** ConvE uses 2D convolutional layers over reshaped and concatenated entity
and relation embeddings. The resulting feature maps are flattened, projected, and
matched against all candidate tail entities. This architecture increases the number
of interaction points and achieves strong results on link prediction benchmarks.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``200``.
- ``use_bias (bool)`` : Whether to use bias. Defaults to ``True``.
- ``input_dropout (float)`` : Input dropout rate. Defaults to ``0.2``.
- ``hidden_dropout (float)`` : Hidden dropout rate. Defaults to ``0.3``.
- ``feature_dropout (float)`` : Feature dropout rate. Defaults to ``0.2``.
- ``hidden_size (int)`` : Hidden layer size. Defaults to ``9728``.
- ``embedding_shape (int)`` : Embedding reshape dimension. Defaults to ``20``.
- ``label_smoothing (float)`` : Label smoothing factor. Defaults to ``0.1``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='ConvE', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [100,200,300]
   input_dropout choice [0.1,0.2,0.3]
   hidden_dropout choice [0.2,0.3,0.4]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

