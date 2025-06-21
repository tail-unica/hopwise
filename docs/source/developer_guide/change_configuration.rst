U Configuration
======================
Here we present how to change or add the configuration object to allow more configuration options, and apply it into hopwise.

La configurazione è gestita dalla classe Config, che viene passata successivamente a tutti gli oggetti successivi come Dataset, Sampler, Dataloader, Trainer e Model.

How does config class work
--------------------------
As you can see, here we have a Config class that is used to manage the configuration of the hopwise framework. The Config class is responsible for loading configuration files, merging them with command line arguments, and providing access to the final configuration dictionary. In the __init__ we have those methods:

.. code:: python

        self.compatibility_settings() # just initialize some numpy datatypes
        self._init_parameters_category() # just set the parameters category based on model type (General, Context, etc.)
        self.yaml_loader = self._build_yaml_loader() # just a basic yaml parser
        self.file_config_dict = self._load_config_files(config_file_list) # read and loads configuration files
        self.variable_config_dict = self._load_variable_config_dict(config_dict) # convert config data to a dictionary
        self.cmd_config_dict = self._load_cmd_line() # load command line arguments
        self._merge_external_config_dict() # merge command line arguments and configuration files

        # retrieve the model and dataset name-
        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model, dataset)
        # load and update configuration dictionary with default yaml configuration files not specified in the config file list
        self._load_internal_config_dict(self.model, self.model_class, self.dataset) # set some basic parameters if not specified in yaml files
        self.final_config_dict = self._get_final_config_dict() # retrieve the final configuration dict
        self._set_default_parameters() # set some default parameters if not specified in yaml files and model INPUT TYPE
        self._init_device() # just set torch device to cpu or cuda
        self._set_torch_dtype() # set torch weight data type
        self._set_train_neg_sample_args() # set negative sampling arguments for training
        self._set_eval_neg_sample_args("valid")
        self._set_eval_neg_sample_args("test")

[TODO] Add a new configuration
-----------------------------
We can update the configuration dict with the `deep_dict_update` function, which allows us to merge two dictionaries recursively. This is useful when we want to update the configuration with new parameters or override existing ones.

.. code:: python
    deep_dict_update(updated_dict, updating_dict)
