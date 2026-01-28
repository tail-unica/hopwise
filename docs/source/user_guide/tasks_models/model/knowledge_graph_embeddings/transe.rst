TransE
===========

Introduction
---------------------

`[paper] <https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`_

**Title:** Translating Embeddings for Modeling Multi-relational Data

**Authors:** Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko

**Abstract:** We consider the problem of embedding entities and relationships
of multi-relational data in low-dimensional vector spaces. Our objective is
to propose a canonical model which is easy to train, contains a reduced number
of parameters and can scale up to very large databases. We propose TransE,
a method which models relationships by interpreting them as translations
operating on the low-dimensional embeddings of the entities.

TransE models relationships as translations in the embedding space:

.. math::
    h + r \\approx t

where :math:`h`, :math:`r`, :math:`t` are embeddings of head entity, relation, and tail entity.

Running with Hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``100``.
- ``margin (float)`` : The margin used in TripletMarginLoss. Defaults to ``1.0``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TransE', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Configuration for Link Prediction:**

.. code:: yaml

   # TransE for link prediction
   task: link_prediction
   embedding_size: 100
   margin: 1.0

   epochs: 500
   train_batch_size: 2048
   learning_rate: 0.001

   metrics: ['Hit', 'MRR']
   topk: [1, 3, 10]

**Configuration for Recommendation:**

TransE can also be used for recommendation by treating user-item interactions
as a special relation in the knowledge graph:

.. code:: yaml

   # TransE for recommendation
   task: recommendation
   embedding_size: 100
   margin: 1.0

   eval_args:
       split: {'RS': [0.8, 0.1, 0.1]}
       mode: full
   metrics: ['Recall', 'NDCG', 'MRR']
   topk: 10

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [50,100,200]
   margin choice [0.5,1.0,2.0]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then run:

.. code:: bash

   hopwise tune --model=TransE --dataset=ml-100k --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

Using TransE Embeddings for Explainable Models
------------------------------------------------

TransE embeddings can be used as pre-trained embeddings for path reasoning models
like PGPR, CAFE, and TPRec. See :doc:`/user_guide/explainability/rl_based_models`
for the complete workflow.

