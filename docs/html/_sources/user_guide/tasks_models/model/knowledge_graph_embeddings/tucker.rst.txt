TuckER
===========

Introduction
---------------------

`[paper] <https://aclanthology.org/D19-1522/>`_

**Title:** TuckER: Tensor Factorization for Knowledge Graph Completion

**Authors:** Ivana Balažević, Carl Allen, Timothy M. Hospedales

**Abstract:** TuckER is based on Tucker decomposition of the binary tensor representation of
knowledge graph triples. It learns entity embeddings, relation embeddings, and a core tensor
for scoring triplets. TuckER is a fully expressive model and can represent any binary relation.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``64``.
- ``input_dropout (float)`` : Input dropout rate. Defaults to ``0.3``.
- ``input_dropout1 (float)`` : First hidden dropout rate. Defaults to ``0.4``.
- ``input_dropout2 (float)`` : Second hidden dropout rate. Defaults to ``0.5``.
- ``label_smoothing (float)`` : Label smoothing factor. Defaults to ``0.1``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='TuckER', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128]
   input_dropout choice [0.2,0.3,0.4]
   label_smoothing choice [0.0,0.1,0.2]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

