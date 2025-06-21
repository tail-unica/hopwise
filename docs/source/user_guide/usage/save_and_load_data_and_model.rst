Save and load data and model
==============================

In this section, we will present how to save and load data and model.

Save data and model
--------------------

When we use the :meth:`~hopwise.quick_start.quick_start.run_hopwise` function mentioned in :doc:`run_hopwise`,
it will save the best model parameters in training process and its corresponding config settings.
If you want to save filtered dataset and split dataloaders,
you can set parameter :attr:`save_dataset` and parameter :attr:`save_dataloaders` to ``True``
to save filtered dataset and split dataloaders.

You can refer to :doc:`../config_settings` for more details about :attr:`save_dataset` and :attr:`save_dataloaders`.


Load data and model
--------------------

If you want to reload the data and model,
you can apply :meth:`~hopwise.quick_start.quick_start.load_data_and_model` to get them.
You can also pass :attr:`dataset_file` and :attr:`dataloader_file` to this function to reload data from file,
which can reduce the time of data filtering and data splitting.

Here we present a typical usage of :meth:`~hopwise.quick_start.quick_start.load_data_and_model`:

.. code:: python3

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='saved/BPR-Aug-21-2021_13-06-00.pth',
    )
    # Here you can replace it by your model path.
    # And you can also pass 'dataset_file' and 'dataloader_file' to this function.
