Use run_hopwise
==========================
We enclose the training and evaluation processes in the api of
:func:`~hopwise.quick_start.quick_start.run_hopwise`,
which is composed of: dataset loading, dataset splitting, model initialization,
model training and model evaluation.

If this process can satisfy your requirement, you can recall this api to use
hopwise.

You can create a python file (e.g., `run.py` ), and write the following code
into the file.

.. code:: python

    from hopwise.quick_start import run_hopwise

    run_hopwise(dataset=dataset, model=model, config_file_list=config_file_list, config_dict=config_dict)

:attr:`dataset` is the name of the data, such as 'ml-100k',
:attr:`model` indicates the model name, such as 'BPR'.

:attr:`config_file_list` indicates the configuration files,
:attr:`config_dict` is the parameter dict.
The two variables are used to config parameters in our toolkit.
If you do not want to use the two variables to config parameters,
please ignore them. In addition, we also support to control parameters
by the command line.

Please refer to :doc:`../configuration` for more details about config settings.

Then execute the following command to run:：

.. code:: bash

    python run.py --[param_name]=[param_value]

`--[param_name]=[param_value]` is the way to control parameters by
the command line.
