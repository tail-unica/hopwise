Pop
===========

Introduction
---------------------

This is a model that records the popularity of items in the dataset and recommend the most popular items to users.

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- No hyper-parameters


**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='Pop', dataset='ml-100k')

And then:

.. code:: bash

   python run.py
