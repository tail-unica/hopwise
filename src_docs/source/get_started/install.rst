Install Hopwise
======================
hopwise can be installed from Source or via PyPI


System requirements
------------------------
hopwise is compatible with the following operating systems:

* Linux ✅
* Windows 10 ✅
* macOS X (untested) ⚠️

Python 3.9, 3.10, 3.11, and 3.12 are supported. Specific dependencies are listed only in the `pyproject.toml <https://github.com/tail-unica/hopwise/blob/main/pyproject.toml>`_ file.

Install with uv from PyPI
-------------------------
To install hopwise with pip, run the following command:

.. code:: bash

    uv pip install hopwise

Some models and functionalities require additional dependencies, such as `torch-geometric` for `KGIN` or `faiss-cpu` for `NCL`.
Here we list and describe the available optional dependencies (a.k.a. "extras") and suggested procedures to install them:

- `pathlm`: for all the language models for KG path reasoning, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[pathlm]

  This will install HuggingFace libraries `transformers`, `datasets` and other utilities (`joblib`, `numba`, `igraph`) as well,
  which are required for the language models.
- `ldiffrec`: for the `LDiffRec` model, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[ldiffrec]

  This will install `kmeans-pytorch` as well, which is required for the `LDiffRec` model.
- `ncl`: for the `NCL` model, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[ncl]

  This will install `faiss-cpu` as well, which is required for the `NCL` model.
- `nncf`: for the `NNCF` model, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[nncf]

  This will install `community` and `python-louvain` as well, which are required for the `NNCF` model.
- `lightgbm`: for the `LightGBM` model, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[lightgbm]

  This will install `lightgbm` as well, which is required for the `LightGBM` model.
- `xgboost`: for the `XGBoost` model, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[xgboost]

  This will install `xgboost` as well, which is required for the `XGBoost` model.
- `pyg`: for the `KGIN`, `KGRec` and `MCCLK` models, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[pyg]

  This will install `torch-geometric` as well, which is required for these models. It is a pure-Python package
  available on PyPI, so no special wheel index is needed.
- `hyper`: to tune the parameters of the models, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[hyper]

  This will install `optuna`, `hyperopt`, `ray`, and `pyplot` as well, which are required for the `tune` interface.
- `tsne`: to visualize KG embeddings with t-SNE, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[tsne]

  This will install `plotly` and `opentsne` as well, which are required for the t-SNE visualization.
- `all-models`: shortcut to install optional dependencies for all models. You can install it with the following command:

  .. code:: bash

    uv pip install hopwise[all-models]

  This will install all the optional dependencies listed above, except `kgin`, which must be installed separately as described above.


Install from Source (Development)
-------------------------
Clone hopwise from GitHub.

.. code:: bash

    git clone https://github.com/tail-unica/hopwise && cd hopwise

Run the following command to install (sync dependencies):

.. code:: bash

    uv sync

Try to run:
-------------------------
To check if you have successfully installed hopwise, you can run:

.. code-block:: bash

    hopwise train

    uv run hopwise train  # alternatively


or create a new python file (e.g., `run.py`), and write the following code:

.. code:: python

    from hopwise.quick_start import run_hopwise

    run_hopwise(model='BPR', dataset='ml-100k')


Then run the following command:

.. code:: bash

    uv run run.py

This will perform the training and test of the BPR model on the ml-100k dataset, and you will obtain some output like:
