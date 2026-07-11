TransE
===========

Introduction
---------------------

`[paper] <https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`_

**Title:** Translating Embeddings for Modeling Multi-relational Data

**Authors:** Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko

**Abstract:** TransE is a method for modeling multi-relational data by interpreting
relationships as translations operating on low-dimensional embeddings of entities.
The key idea is that if a triplet (head, relation, tail) holds, then the embedding
of the tail entity should be close to the embedding of the head entity plus some
vector that depends on the relationship. This simple yet effective approach learns
embeddings such that h + r ≈ t when (h, r, t) is a valid triplet.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``100``.
- ``margin (float)`` : The margin used in the TripletMarginLoss. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TransE', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Notes:**

- TransE supports both recommendation and link prediction tasks.
- Requires knowledge graph data (`.kg` and `.link` files).

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [50,100,200]
   margin choice [0.5,1.0,2.0]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

