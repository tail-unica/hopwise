hopwise.quick_start
===================

.. py:module:: hopwise.quick_start


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/quick_start/quick_start/index


Functions
---------

.. autoapisummary::

   hopwise.quick_start.load_data_and_model
   hopwise.quick_start.objective_function
   hopwise.quick_start.run
   hopwise.quick_start.run_hopwise
   hopwise.quick_start.run_hopwises


Package Contents
----------------

.. py:function:: load_data_and_model(model_file, load_only_data=False, updating_config=None)

   Load filtered dataset, split dataloaders and saved model.

   :param model_file: The path of saved model file.
   :type model_file: str
   :param load_only_data: Whether to load only the dataset and dataloaders without the model.
                          Defaults to ``False``.
   :type load_only_data: bool, optional
   :param updating_config: A Config object to update the config parameters loaded from checkpoint.
                           Defaults to ``None``.
   :type updating_config: Config, optional

   :returns:     - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
                 - model (AbstractRecommender): The model load from :attr:`model_file`.
                 - dataset (Dataset): The filtered dataset.
                 - train_data (AbstractDataLoader): The dataloader for training.
                 - valid_data (AbstractDataLoader): The dataloader for validation.
                 - test_data (AbstractDataLoader): The dataloader for testing.
   :rtype: tuple


.. py:function:: objective_function(config_dict=None, config_file_list=None, saved=True, show_progress=False, callback_fn=None)

   The default objective_function used in HyperTuning

   :param config_dict: Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
   :type config_dict: dict, optional
   :param config_file_list: Config files used to modify experiment parameters. Defaults to ``None``.
   :type config_file_list: list, optional
   :param saved: Whether to save the model. Defaults to ``True``.
   :type saved: bool, optional


.. py:function:: run(model, dataset, run='train', checkpoint=None, config_file_list=None, config_dict=None, saved=True, nproc=1, world_size=-1, ip='localhost', port='5678', group_offset=0)

.. py:function:: run_hopwise(model=None, dataset=None, run='train', checkpoint=None, config_file_list=None, config_dict=None, saved=True, queue=None)

   A fast running api, which includes the complete process of
   training and testing a model on a specified dataset

   :param model: Model name. Defaults to ``None``.
   :type model: str, optional
   :param dataset: Dataset name. Defaults to ``None``.
   :type dataset: str, optional
   :param run: The running mode, 'train' or 'evaluate'. Defaults to ``'train'``.
   :type run: str, optional
   :param checkpoint: The path of the saved model file. Defaults to ``None``.
   :type checkpoint: str, optional
   :param config_file_list: Config files used to modify experiment parameters. Defaults to ``None``.
   :type config_file_list: list, optional
   :param config_dict: Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
   :type config_dict: dict, optional
   :param saved: Whether to save the model. Defaults to ``True``.
   :type saved: bool, optional
   :param queue: The queue used to pass the result to the main process. Defaults to ``None``.
   :type queue: torch.multiprocessing.Queue, optional


.. py:function:: run_hopwises(rank, *args)

