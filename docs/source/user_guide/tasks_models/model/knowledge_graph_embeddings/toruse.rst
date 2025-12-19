TorusE
===========

Introduction
---------------------

`[paper] <https://ojs.aaai.org/index.php/AAAI/article/view/11538>`_

**Title:** TorusE: Knowledge Graph Embedding on a Lie Group

**Authors:** Takuma Ebisu, Ryutaro Ichise

**Abstract:** TorusE extends TransE by embedding entities and relations on a torus
(a Lie group). This allows the model to avoid the regularization issues that
TransE faces and provides a more principled way of handling the translation-based
scoring function. The torus structure naturally handles the periodicity.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TorusE', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

