Use run_hopwise()
==========================
We enclose the training and evaluation processes in the api of
:func:`~hopwise.quick_start.quick_start.run_hopwise`,
which is composed of: dataset loading, dataset splitting, model initialization,
model training, model evaluation, checkpoint loading and type of execution (only evaluation on checkpoint or full training).

You can create a python file (e.g., `run.py` ), and write the following code
into the file.

.. code:: python

    from hopwise.quick_start import run_hopwise

    run_hopwise(dataset=dataset, model=model, config_file_list=config_file_list, config_dict=config_dict)

:attr:`dataset` is the name of the data, such as 'ml-100k',

:attr:`model` indicates the model name, such as 'BPR'.

:attr:`config_file_list` indicates the configuration files (e.g., `[file1.yaml,file2.yaml,..]` ),

:attr:`config_dict` is the parameter dict.

:attr:`checkpoint` full path of the torch .pth checkpoint file

:attr:`run` whether to run the training (default ``run='train'``) or only the evaluation (``run='evaluate'``) given a checkpoint.

.. important::

    The difference between ``config_dict`` and ``config_file_list`` is that with config_dict you can create a Python dictionary containing the arguments you want to pass, while with ``config_file_list`` you can create multiple ``.yaml`` files where you set different parameters. The order of precedence is ``config_file_list config_dict``, so if a parameter defined in ``config_file_list`` is also present in config_dict, it will be overridden.
