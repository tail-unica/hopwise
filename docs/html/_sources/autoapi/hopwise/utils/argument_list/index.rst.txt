hopwise.utils.argument_list
===========================

.. py:module:: hopwise.utils.argument_list


Attributes
----------

.. autoapisummary::

   hopwise.utils.argument_list.general_arguments
   hopwise.utils.argument_list.training_arguments
   hopwise.utils.argument_list.evaluation_arguments
   hopwise.utils.argument_list.dataset_arguments


Module Contents
---------------

.. py:data:: general_arguments
   :value: ['gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir',...


.. py:data:: training_arguments
   :value: ['epochs', 'train_batch_size', 'learner', 'learning_rate', 'train_neg_sample_args', 'eval_step',...


.. py:data:: evaluation_arguments
   :value: ['eval_args', 'repeatable', 'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',...


.. py:data:: dataset_arguments
   :value: ['field_separator', 'seq_separator', 'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD',...


