ConvKB
===========

Introduction
---------------------

`[paper] <https://aclanthology.org/N18-2053/>`_

**Title:** A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network

**Authors:** Dai Quoc Nguyen, Tu Dinh Nguyen, Dat Quoc Nguyen, Dinh Phung

**Abstract:** ConvKB applies convolutional neural networks directly on the concatenation
of entity and relation embeddings. The model captures global relationships among
the embeddings through convolution filters and achieves state-of-the-art results
on link prediction tasks.


Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of entities and relations. Defaults to ``50``.
- ``out_channels (int)`` : Number of output channels. Defaults to ``64``.
- ``dropout_prob (float)`` : Dropout probability. Defaults to ``0``.
- ``kernel_size (int)`` : Kernel size for convolution. Defaults to ``1``.
- ``lambda (float)`` : Regularization weight. Defaults to ``0.2``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='ConvKB', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [50,100,200]
   out_channels choice [32,64,128]
   dropout_prob choice [0,0.1,0.2]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

