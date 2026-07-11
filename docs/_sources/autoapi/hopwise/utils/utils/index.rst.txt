hopwise.utils.utils
===================

.. py:module:: hopwise.utils.utils

.. autoapi-nested-parse::

   hopwise.utils.utils
   ################################



Classes
-------

.. autoapisummary::

   hopwise.utils.utils.GenerationOutputs


Functions
---------

.. autoapisummary::

   hopwise.utils.utils.get_local_time
   hopwise.utils.utils.ensure_dir
   hopwise.utils.utils.deep_dict_update
   hopwise.utils.utils.get_model
   hopwise.utils.utils.get_trainer
   hopwise.utils.utils.early_stopping
   hopwise.utils.utils.calculate_valid_score
   hopwise.utils.utils.dict2str
   hopwise.utils.utils.init_seed
   hopwise.utils.utils.get_tensorboard
   hopwise.utils.utils.get_gpu_usage
   hopwise.utils.utils.get_flops
   hopwise.utils.utils.list_to_latex
   hopwise.utils.utils.get_environment
   hopwise.utils.utils.get_sequence_postprocessor
   hopwise.utils.utils.get_logits_processor


Module Contents
---------------

.. py:function:: get_local_time()

   Get current time

   :returns: current time
   :rtype: str


.. py:function:: ensure_dir(dir_path)

   Make sure the directory exists, if it does not exist, create it

   :param dir_path: directory path
   :type dir_path: str


.. py:function:: deep_dict_update(updated_dict, updating_dict)

.. py:function:: get_model(model_name)

   Automatically select model class based on model name

   :param model_name: model name
   :type model_name: str

   :returns: model class
   :rtype: Recommender


.. py:function:: get_trainer(model_type, model_name)

   Automatically select trainer class based on model type and model name

   :param model_type: model type
   :type model_type: ModelType
   :param model_name: model name
   :type model_name: str

   :returns: trainer class
   :rtype: Trainer


.. py:function:: early_stopping(value, best, cur_step, max_step, bigger=True)

   validation-based early stopping

   :param value: current result
   :type value: float
   :param best: best result
   :type best: float
   :param cur_step: the number of consecutive steps that did not exceed the best result
   :type cur_step: int
   :param max_step: threshold steps for stopping
   :type max_step: int
   :param bigger: whether the bigger the better
   :type bigger: bool, optional

   :returns:

             - float,
               best result after this step
             - int,
               the number of consecutive steps that did not exceed the best result after this step
             - bool,
               whether to stop
             - bool,
               whether to update
   :rtype: tuple


.. py:function:: calculate_valid_score(valid_result, valid_metric=None)

   Return valid score from valid result

   :param valid_result: valid result
   :type valid_result: dict
   :param valid_metric: the selected metric in valid result for valid score
   :type valid_metric: str, optional

   :returns: valid score
   :rtype: float


.. py:function:: dict2str(result_dict)

   Convert result dict to str

   :param result_dict: result dict
   :type result_dict: dict

   :returns: result str
   :rtype: str


.. py:function:: init_seed(seed, reproducibility)

   Init random seed for random functions in numpy, torch, cuda and cudnn

   :param seed: random seed
   :type seed: int
   :param reproducibility: Whether to require reproducibility
   :type reproducibility: bool


.. py:function:: get_tensorboard(logger)

   Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
   visualization within the TensorBoard UI.
   For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

   :param logger: its output filename is used to name the SummaryWriter's log_dir.
                  If the filename is not available, we will name the log_dir according to the current time.

   :returns: it will write out events and summaries to the event file.
   :rtype: SummaryWriter


.. py:function:: get_gpu_usage(device=None)

   Return the reserved memory and total memory of given device in a string.

   :param device: cuda.device. It is the device that the model run on.

   :returns: it contains the info about reserved memory and total memory of given device.
   :rtype: str


.. py:function:: get_flops(model, dataset, device, logger, transform, verbose=False)

   Given a model and dataset to the model, compute the per-operator flops
   of the given model.

   :param model: the model to compute flop counts.
   :param dataset: dataset that are passed to `model` to count flops.
   :param device: cuda.device. It is the device that the model run on.
   :param verbose: whether to print information of modules.

   :returns: the number of flops for each operation.
   :rtype: total_ops


.. py:function:: list_to_latex(convert_list, bigger_flag=True, subset_columns=[])

.. py:function:: get_environment(config)

.. py:function:: get_sequence_postprocessor(postprocessor_name)

.. py:function:: get_logits_processor(model_name)

.. py:class:: GenerationOutputs

   Bases: :py:obj:`dict`


   Dataclass to hold the outputs of the generation process.

   .. attribute:: sequences

      The generated sequences.

      :type: torch.Tensor

   .. attribute:: scores

      The scores for each generated token.

      :type: torch.Tensor


   .. py:attribute:: sequences
      :type:  torch.Tensor


   .. py:attribute:: scores
      :type:  torch.Tensor


   .. py:method:: __post_init__()


