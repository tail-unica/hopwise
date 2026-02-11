DistMult
===========

Introduction
---------------------

`[paper] <https://arxiv.org/abs/1412.6575>`_

**Title:** Embedding Entities and Relations for Learning and Inference in Knowledge Bases

**Authors:** Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng

**Abstract:** DistMult is a simple bilinear model for knowledge graph embedding. It represents
each relation as a diagonal matrix and scores triplets using a bilinear function.
Despite its simplicity, DistMult achieves competitive results on link prediction tasks.
However, it can only model symmetric relations.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``margin (float)`` : The margin used in the MarginRankingLoss. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='DistMult', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128,256]
   margin choice [0.5,1.0,2.0]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

