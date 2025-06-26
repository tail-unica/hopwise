Install Hopwise
======================
hopwise can be installed from Source or via PyPI


System requirements
------------------------
hopwise is compatible with the following operating systems:

* Linux (we currently use hopwise in a ubuntu server)
* Windows 10
* macOS X

Python 3.9, 3.10, 3.11, view the pyproject.toml file for the other important dependencies.


Install with uv
-------------------------
To install hopwise with pip, only the following command is needed:

.. code:: bash

    uv pip install hopwise


Install from Source
-------------------------
Download the Source files from GitHub.

.. code:: bash

    git clone https://github.com/tail-unica/hopwise && cd hopwise

Run the following command to install:

.. code:: bash

    uv pip install -e . --verbose

Try to run:
-------------------------
To check if you have successfully installed the RecBole, you can run:

.. code:: bash

    hopwise train


or create a new python file (e.g., `run.py`), and write the following code:

.. code:: python

    from hopwise.quick_start import run_hopwise

    run_hopwise(model='BPR', dataset='ml-100k')


Then run the following command:

.. code:: bash

    python run.py

This will perform the training and test of the BPR model on the ml-100k dataset, and you will obtain some output like: