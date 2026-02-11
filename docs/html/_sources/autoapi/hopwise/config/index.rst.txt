hopwise.config
==============

.. py:module:: hopwise.config


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/config/configurator/index


Classes
-------

.. autoapisummary::

   hopwise.config.Config


Package Contents
----------------

.. py:class:: Config(model=None, dataset=None, config_file_list=None, config_dict=None)

   Configurator module that load the defined parameters.

   Configurator module will first load the default parameters from the fixed properties in Hopwise and then
   load parameters from the external input.

   External input supports three kind of forms: config file, command line and parameter dictionaries.

   - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
     e.g. a config file is 'example.yaml', the content is:

       learning_rate: 0.001

       train_batch_size: 2048

   - command line: It should be in the format as '---learning_rate=0.001'

   - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
     e.g. config_dict = {'learning_rate': 0.001}

   Configuration module allows the above three kind of external input format to be used together,
   the priority order is as following:

   command line > parameter dictionaries > config file

   e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
   learning_rate=0.03 in parameter dictionaries.

   Finally the learning_rate is equal to 0.02.


   .. py:attribute:: NESTED_KEY_SEPARATOR
      :value: '.'



   .. py:attribute:: yaml_loader


   .. py:attribute:: file_config_dict


   .. py:attribute:: variable_config_dict


   .. py:attribute:: cmd_config_dict


   .. py:attribute:: final_config_dict


   .. py:method:: _init_parameters_category()


   .. py:method:: _build_yaml_loader()


   .. py:method:: _convert_config_dict(config_dict)

      This function convert the str parameters to their original type.



   .. py:method:: _load_config_files(file_list)


   .. py:method:: _load_variable_config_dict(config_dict)


   .. py:method:: _load_cmd_line()

      Read parameters from command line and convert it to str.



   .. py:method:: _merge_external_config_dict()


   .. py:method:: _get_model_and_dataset(model, dataset)


   .. py:method:: _update_internal_config_dict(file)


   .. py:method:: _load_internal_config_dict(model, model_class, dataset)


   .. py:method:: _get_final_config_dict()


   .. py:method:: _set_default_parameters()


   .. py:method:: _init_device()


   .. py:method:: _set_train_neg_sample_args()


   .. py:method:: _set_eval_neg_sample_args(phase: Literal['valid', 'test'])


   .. py:method:: _set_torch_dtype()

      Convert a string dtype to a torch dtype.



   .. py:method:: _set_env_behavior()

      Set behavior of utilities or similar libraries based on environment variables.



   .. py:method:: __setitem__(key, value)


   .. py:method:: __getattr__(item)


   .. py:method:: __getitem__(item)


   .. py:method:: __contains__(key)


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: compatibility_settings()


