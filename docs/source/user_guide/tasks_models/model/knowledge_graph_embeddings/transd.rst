TransD
===========

Introduction
---------------------

`[paper] <https://aclanthology.org/P15-1067/>`_

**Title:** Knowledge Graph Embedding via Dynamic Mapping Matrix

**Authors:** Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao

**Abstract:** TransD constructs a dynamic mapping matrix for each entity-relation pair,
replacing the static projection matrix in TransR. This reduces the number of parameters
and allows for more efficient learning while capturing the diversity of entities and relations.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``margin (float)`` : The margin used in the TripletMarginLoss. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TransD', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128]
   margin choice [0.5,1.0,2.0]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

