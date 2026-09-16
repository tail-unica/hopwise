Item Cold Start
===================


Currently available for General Recommenders, where through the .yaml configuration file the user specifies the items that can be recommended during the test set, while everything else is masked.

For example, in BPR.yaml file you set:

.. code:: bash

    benchmark_filename: ["train", "test"]
    benchmark_item_filename: ["train", "test"]

This means that in the dataset folder xxx/yyy, where xxx is given by the data_path argument, you must have ``dataset.train.inter``, ``dataset.valid.inter``, ``dataset.train.item``, ``dataset.test.item``. With the benchmark filename, we specify the train and test splits, while the .item files contain the original dataset IDs of the items that can be predicted for that split. Therefore, during the prediction phase in inference, all items not present in the .test.item file are masked.


``dataset_name.test.item`` example file:

=============
item_id:token
=============
78
9318
788
...
=============