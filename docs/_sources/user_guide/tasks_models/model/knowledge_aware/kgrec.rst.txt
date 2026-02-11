KGRec
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3580305.3599400>`_

**Title:** Knowledge Graph Self-Supervised Rationalization for Recommendation

**Authors:** Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang

**Abstract:** In this paper, we introduce a new self-supervised rationalization
method, called KGRec, for knowledge-aware recommender systems. To effectively
identify informative knowledge connections, we propose an attentive knowledge
rationalization mechanism that generates rational scores for knowledge triplets.
With these scores, KGRec integrates generative and contrastive self-supervised
tasks for recommendation through rational masking.

To highlight rationales in the knowledge graph, we design a novel generative task
in the form of masking-reconstructing. By masking important knowledge with high
rational scores, KGRec is trained to rebuild and highlight useful knowledge
connections that serve as rationales. To further rationalize the effect of
collaborative interactions on knowledge graph learning, we introduce a contrastive
learning task that aligns signals from knowledge and user-item interaction views.
To ensure noise-resistant contrasting, potential noisy edges in both graphs judged
by the rational scores are masked.

Extensive experiments on three real-world datasets demonstrate that KGRec
outperforms state-of-the-art methods.

Running with hopwise
-------------------------

**Model Hyper-Parameters:**

- ``embedding_size (int)`` : The embedding size of users, items, entities and relations. Defaults to ``64``.
- ``reg_weight (float)`` : The L2 regularization weight. Defaults to ``1e-5``.
- ``context_hops (int)`` : The number of context hops in GCN layer. Defaults to ``2``.
- ``node_dropout_rate (float)`` : The node dropout rate in GCN layer. Defaults to ``0.5``.
- ``mess_dropout_rate (float)`` : The message dropout rate in GCN layer. Defaults to ``0.1``.
- ``mae_coef (float)`` : The masked autoencoder loss coefficient. Defaults to ``0.1``.
- ``mae_msize (int)`` : The size of the masked edge set for autoencoder reconstruction. Defaults to ``256``.
- ``cl_coef (float)`` : The contrastive loss coefficient. Defaults to ``0.01``.
- ``cl_tau (float)`` : The temperature in contrastive loss. Defaults to ``1.0``.
- ``cl_drop (float)`` : The graph dropout rate for contrastive learning. Defaults to ``0.5``.
- ``samp_func (str)`` : The type of sampling function. Defaults to ``'torch'``. Range in ``['torch', 'np']``.


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from hopwise.quick_start import run_hopwise

   run_hopwise(model='KGRec', dataset='ml-100k')

And then:

.. code:: bash

   python run.py

**Notes:**

- KGRec requires ``torch-scatter`` for efficient scatter operations on GPU. Please install it separately following the `PyTorch Geometric installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_.

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   embedding_size choice [32,64,128]
   context_hops choice [1,2,3]
   reg_weight choice [1e-4,1e-5,1e-6]
   mess_dropout_rate choice [0.1,0.2,0.3]
   mae_coef choice [0.05,0.1,0.2]
   cl_coef choice [0.005,0.01,0.02]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of hopwise (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	hopwise tune --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`/user_guide/usage/parameter_tuning`.

