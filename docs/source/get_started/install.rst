Install Hopwise
======================
hopwise can be installed from Source or via PyPI


System requirements
------------------------
hopwise is compatible with the following operating systems:

* Linux ✅
* Windows 10 (partially supported due to some issues with `dgl`) ⚠️
* macOS X (untested) ⚠️

Python 3.9, 3.10, 3.11, 3.12 are supported. Specific dependencies are solely reported in the [pyproject.toml](https://github.com/tail-unica/hopwise/blob/main/pyproject.toml) file.


Install with uv from PyPI
-------------------------
To install hopwise with pip, run the following command:

.. code:: bash

    uv pip install hopwise

Some models and functionalities require additional dependencies, such as `torch-scatter` for `KGIN` or `faiss-cpu` for `NCL`.
Here we list and describe the available optional dependencies (a.k.a. "extras") and suggested procedures to install them:

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
- `kgat`: for the `KGAT` model, installation with `hopwise[kgat]` is not supported, as `dgl` causes some issues with `torch`.
    It must be then installed separately preventing the installation of additional dependencies (which unnecessarily overwrite the `torch` version) with the following command:

  .. code:: bash

    uv pip install dgl>=2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

  Please check out the `DGL installation guide <https://www.dgl.ai/pages/start.html>`__ for more details and for the URLs of other flat indexes for different CUDA/CPU versions.
- `kgin`: for the `KGIN` model, installation with `hopwise[kgin]` is not supported, as `torch-scatter` needs to be installed separately.
  You can install it with the following command (assuming you have PyTorch 2.7.* and CUDA 12.8 installed):

  .. code:: bash

    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

  Please check out the `torch-scatter` `GitHub webpage <https://github.com/rusty1s/pytorch_geometric>`__ for more details on how to install it correctly for your system.
- `hyper`: to tune the parameters of the models, you can install it with the following command:

  .. code:: bash

    uv pip install hopwise[hyper]

  This will install `optuna`, `hyperopt`, `ray` as well, which are required for the `tune` interface.
- `all-models`: shortcut to install optional dependencies for all models. You can install it with the following command:

  .. code:: bash

    uv pip install hopwise[all-models]

  This will install all the optional dependencies listed above, except `kgin`, which must be installed separately as described above.
  Be aware that this is mainly intended for development purposes, as the `dgl` dependency of `kgat` will probably conflict with the `torch` version installed by `hopwise`.


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