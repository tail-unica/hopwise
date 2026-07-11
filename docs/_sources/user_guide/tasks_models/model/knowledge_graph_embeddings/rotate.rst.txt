RotatE
===========

Introduction
---------------------

`[paper] <https://openreview.net/forum?id=HkgEQnRqYQ>`_

**Title:** RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

**Authors:** Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, Jian Tang

**Abstract:** RotatE models relations as rotations from head to tail entities in complex vector space.
This allows it to model and infer relation patterns including symmetry, antisymmetry, inversion,
and composition. The key idea is that each relation is represented as an element-wise rotation
from the head entity to the tail entity in complex space.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``margin (float)`` : The margin used in the scoring function. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='RotatE', dataset='ml-100k')

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

