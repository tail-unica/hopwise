Analogy
===========

Introduction
---------------------

`[paper] <https://proceedings.mlr.press/v70/liu17d.html>`_

**Title:** Analogical Inference for Multi-Relational Embeddings

**Authors:** Hanxiao Liu, Yuexin Wu, Yiming Yang

**Abstract:** Analogy learns entity and relation embeddings that support analogical inference.
The model is designed to capture analogical structures in knowledge graphs, where relations
between entity pairs can be characterized by analogies (e.g., king:queen :: man:woman).


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``scalar_share (float)`` : Scalar share parameter. Defaults to ``0.5``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='Analogy', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128]
   scalar_share choice [0.3,0.5,0.7]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

