TransR
===========

Introduction
---------------------

`[paper] <https://ojs.aaai.org/index.php/AAAI/article/view/9491>`_

**Title:** Learning Entity and Relation Embeddings for Knowledge Graph Completion

**Authors:** Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu

**Abstract:** TransR models entities and relations in distinct spaces. Each relation has a
projection matrix that projects entity embeddings from entity space to relation space.
This allows for more flexible modeling of complex relations compared to TransE and TransH.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``margin (float)`` : The margin used in the TripletMarginLoss. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TransR', dataset='ml-100k')

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

