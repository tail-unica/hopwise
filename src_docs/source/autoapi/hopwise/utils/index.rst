hopwise.utils
=============

.. py:module:: hopwise.utils


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/utils/argument_list/index
   /autoapi/hopwise/utils/case_study/index
   /autoapi/hopwise/utils/enum_type/index
   /autoapi/hopwise/utils/logger/index
   /autoapi/hopwise/utils/url/index
   /autoapi/hopwise/utils/utils/index
   /autoapi/hopwise/utils/wandblogger/index


Attributes
----------

.. autoapisummary::

   hopwise.utils.general_arguments
   hopwise.utils.training_arguments
   hopwise.utils.evaluation_arguments
   hopwise.utils.dataset_arguments


Classes
-------

.. autoapisummary::

   hopwise.utils.ModelType
   hopwise.utils.KGDataLoaderState
   hopwise.utils.EvaluatorType
   hopwise.utils.InputType
   hopwise.utils.FeatureType
   hopwise.utils.FeatureSource
   hopwise.utils.PathLanguageModelingTokenType
   hopwise.utils.GenerationOutputs
   hopwise.utils.WandbLogger


Functions
---------

.. autoapisummary::

   hopwise.utils.init_logger
   hopwise.utils.progress_bar
   hopwise.utils.set_color
   hopwise.utils.calculate_valid_score
   hopwise.utils.deep_dict_update
   hopwise.utils.dict2str
   hopwise.utils.early_stopping
   hopwise.utils.ensure_dir
   hopwise.utils.get_environment
   hopwise.utils.get_flops
   hopwise.utils.get_gpu_usage
   hopwise.utils.get_local_time
   hopwise.utils.get_logits_processor
   hopwise.utils.get_model
   hopwise.utils.get_sequence_postprocessor
   hopwise.utils.get_tensorboard
   hopwise.utils.get_trainer
   hopwise.utils.init_seed
   hopwise.utils.list_to_latex


Package Contents
----------------

.. py:data:: general_arguments
   :value: ['gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir',...


.. py:data:: training_arguments
   :value: ['epochs', 'train_batch_size', 'learner', 'learning_rate', 'train_neg_sample_args', 'eval_step',...


.. py:data:: evaluation_arguments
   :value: ['eval_args', 'repeatable', 'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',...


.. py:data:: dataset_arguments
   :value: ['field_separator', 'seq_separator', 'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD',...


.. py:class:: ModelType

   Bases: :py:obj:`enum.Enum`


   Type of models.

   - ``GENERAL``: General Recommendation
   - ``SEQUENTIAL``: Sequential Recommendation
   - ``CONTEXT``: Context-aware Recommendation
   - ``KNOWLEDGE``: Knowledge-based Recommendation
   - ``PATH_LANGUAGE_MODELING``: Path Language Modeling Recommendation


   .. py:attribute:: GENERAL
      :value: 1



   .. py:attribute:: SEQUENTIAL
      :value: 2



   .. py:attribute:: CONTEXT
      :value: 3



   .. py:attribute:: KNOWLEDGE
      :value: 4



   .. py:attribute:: TRADITIONAL
      :value: 5



   .. py:attribute:: DECISIONTREE
      :value: 6



   .. py:attribute:: PATH_LANGUAGE_MODELING
      :value: 7



.. py:class:: KGDataLoaderState

   Bases: :py:obj:`enum.Enum`


   States for Knowledge-based DataLoader.

   - ``RSKG``: Return both knowledge graph information and user-item interaction information.
   - ``RS``: Only return the user-item interaction.
   - ``KG``: Only return the triplets with negative examples in a knowledge graph.


   .. py:attribute:: RSKG
      :value: 1



   .. py:attribute:: RS
      :value: 2



   .. py:attribute:: KG
      :value: 3



.. py:class:: EvaluatorType

   Bases: :py:obj:`enum.Enum`


   Type for evaluation metrics.

   - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
   - ``VALUE``: Value-based metrics like AUC, etc.


   .. py:attribute:: RANKING
      :value: 1



   .. py:attribute:: VALUE
      :value: 2



.. py:class:: InputType

   Bases: :py:obj:`enum.Enum`


   Type of Models' input.

   - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
   - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
   - ``LISTWISE``: List-wise input, like ``uid, [iid1, iid2, ...]``.
   - ``PATHWISE``: KG Path-wise input, like ``uid, pos_iid, eid1, eid2, next_pos_iid``.
   - ``USERWISE``: User-wise input, like ``uid0, uid1, ...., uidn``.


   .. py:attribute:: POINTWISE
      :value: 1



   .. py:attribute:: PAIRWISE
      :value: 2



   .. py:attribute:: LISTWISE
      :value: 3



   .. py:attribute:: PATHWISE
      :value: 4



   .. py:attribute:: USERWISE
      :value: 5



.. py:class:: FeatureType

   Bases: :py:obj:`enum.Enum`


   Type of features.

   - ``TOKEN``: Token features like user_id and item_id.
   - ``FLOAT``: Float features like rating and timestamp.
   - ``TOKEN_SEQ``: Token sequence features like review.
   - ``FLOAT_SEQ``: Float sequence features like pretrained vector.


   .. py:attribute:: TOKEN
      :value: 'token'



   .. py:attribute:: FLOAT
      :value: 'float'



   .. py:attribute:: TOKEN_SEQ
      :value: 'token_seq'



   .. py:attribute:: FLOAT_SEQ
      :value: 'float_seq'



.. py:class:: FeatureSource

   Bases: :py:obj:`enum.Enum`


   Source of features.

   - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
   - ``USER``: Features from ``.user`` (other than ``user_id``).
   - ``ITEM``: Features from ``.item`` (other than ``item_id``).
   - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
   - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
   - ``KG``: Features from ``.kg``.
   - ``NET``: Features from ``.net``.


   .. py:attribute:: INTERACTION
      :value: 'inter'



   .. py:attribute:: USER
      :value: 'user'



   .. py:attribute:: ITEM
      :value: 'item'



   .. py:attribute:: USER_ID
      :value: 'user_id'



   .. py:attribute:: ITEM_ID
      :value: 'item_id'



   .. py:attribute:: KG
      :value: 'kg'



   .. py:attribute:: NET
      :value: 'net'



.. py:class:: PathLanguageModelingTokenType(token, token_id)

   Bases: :py:obj:`enum.Enum`


   Type of tokens in paths for Path Language Modeling.

   - ``SPECIAL``: Special tokens, like start and end of a path.
   - ``ENTITY``: Entity tokens.
   - ``RELATION``: Relation tokens.
   - ``USER``: User tokens.
   - ``ITEM``: Item tokens.


   .. py:attribute:: SPECIAL
      :value: ('S', 0)



   .. py:attribute:: ENTITY
      :value: ('E', 1)



   .. py:attribute:: RELATION
      :value: ('R', 2)



   .. py:attribute:: USER
      :value: ('U', 3)



   .. py:attribute:: ITEM
      :value: ('I', 4)



   .. py:attribute:: token


   .. py:attribute:: token_id


   .. py:method:: __str__()


.. py:function:: init_logger(config)

   A logger that can show a message on standard output and write it into the
   file named `filename` simultaneously.
   All the message that you want to log MUST be str.

   :param config: An instance object of Config, used to record parameter information.
   :type config: Config

   .. rubric:: Example

   >>> logger = logging.getLogger(config)
   >>> logger.debug(train_state)
   >>> logger.info(train_result)


.. py:function:: progress_bar(*args, **kwargs)

.. py:function:: set_color(log, color, highlight=True, progress=False)

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


.. py:function:: calculate_valid_score(valid_result, valid_metric=None)

   Return valid score from valid result

   :param valid_result: valid result
   :type valid_result: dict
   :param valid_metric: the selected metric in valid result for valid score
   :type valid_metric: str, optional

   :returns: valid score
   :rtype: float


.. py:function:: deep_dict_update(updated_dict, updating_dict)

.. py:function:: dict2str(result_dict)

   Convert result dict to str

   :param result_dict: result dict
   :type result_dict: dict

   :returns: result str
   :rtype: str


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


.. py:function:: ensure_dir(dir_path)

   Make sure the directory exists, if it does not exist, create it

   :param dir_path: directory path
   :type dir_path: str


.. py:function:: get_environment(config)

.. py:function:: get_flops(model, dataset, device, logger, transform, verbose=False)

   Given a model and dataset to the model, compute the per-operator flops
   of the given model.

   :param model: the model to compute flop counts.
   :param dataset: dataset that are passed to `model` to count flops.
   :param device: cuda.device. It is the device that the model run on.
   :param verbose: whether to print information of modules.

   :returns: the number of flops for each operation.
   :rtype: total_ops


.. py:function:: get_gpu_usage(device=None)

   Return the reserved memory and total memory of given device in a string.

   :param device: cuda.device. It is the device that the model run on.

   :returns: it contains the info about reserved memory and total memory of given device.
   :rtype: str


.. py:function:: get_local_time()

   Get current time

   :returns: current time
   :rtype: str


.. py:function:: get_logits_processor(model_name)

.. py:function:: get_model(model_name)

   Automatically select model class based on model name

   :param model_name: model name
   :type model_name: str

   :returns: model class
   :rtype: Recommender


.. py:function:: get_sequence_postprocessor(postprocessor_name)

.. py:function:: get_tensorboard(logger)

   Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
   visualization within the TensorBoard UI.
   For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

   :param logger: its output filename is used to name the SummaryWriter's log_dir.
                  If the filename is not available, we will name the log_dir according to the current time.

   :returns: it will write out events and summaries to the event file.
   :rtype: SummaryWriter


.. py:function:: get_trainer(model_type, model_name)

   Automatically select trainer class based on model type and model name

   :param model_type: model type
   :type model_type: ModelType
   :param model_name: model name
   :type model_name: str

   :returns: trainer class
   :rtype: Trainer


.. py:function:: init_seed(seed, reproducibility)

   Init random seed for random functions in numpy, torch, cuda and cudnn

   :param seed: random seed
   :type seed: int
   :param reproducibility: Whether to require reproducibility
   :type reproducibility: bool


.. py:function:: list_to_latex(convert_list, bigger_flag=True, subset_columns=[])

.. py:class:: WandbLogger(config)

   WandbLogger to log metrics to Weights and Biases.


   .. py:attribute:: config


   .. py:attribute:: log_wandb


   .. py:method:: setup()


   .. py:method:: log_metrics(metrics, head='train', commit=True)


   .. py:method:: log_eval_metrics(metrics, head='eval')


   .. py:method:: _set_steps()


   .. py:method:: _add_head_to_metrics(metrics, head)


