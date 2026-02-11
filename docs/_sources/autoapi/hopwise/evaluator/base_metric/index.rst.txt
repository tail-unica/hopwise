hopwise.evaluator.base_metric
=============================

.. py:module:: hopwise.evaluator.base_metric

.. autoapi-nested-parse::

   hopwise.evaluator.abstract_metric
   #####################################



Classes
-------

.. autoapisummary::

   hopwise.evaluator.base_metric.AbstractMetric
   hopwise.evaluator.base_metric.TopkMetric
   hopwise.evaluator.base_metric.LossMetric
   hopwise.evaluator.base_metric.ConsumerTopKMetric
   hopwise.evaluator.base_metric.PathQualityMetric


Module Contents
---------------

.. py:class:: AbstractMetric(config)

   :class:`AbstractMetric` is the base object of all metrics. If you want to
       implement a metric, you should inherit this class.

   :param config: the config of evaluator.
   :type config: Config


   .. py:attribute:: smaller
      :value: False



   .. py:attribute:: metric_need
      :value: []



   .. py:attribute:: decimal_place


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:


      Automatically extend parent's metric_need if subclass defines metric_need.



   .. py:method:: calculate_metric(dataobject)
      :abstractmethod:


      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



.. py:class:: TopkMetric(config)

   Bases: :py:obj:`AbstractMetric`


   :class:`TopkMetric` is a base object of top-k metrics. If you want to
   implement an top-k metric, you can inherit this class.

   :param config: The config of evaluator.
   :type config: Config


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.topk']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the bool matrix indicating whether the corresponding item is positive
      and number of positive items for each user.



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



   .. py:method:: metric_info(pos_index, pos_len=None)
      :abstractmethod:


      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: LossMetric(config)

   Bases: :py:obj:`AbstractMetric`


   :class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
   implement an loss based metric, you can inherit this class.

   :param config: The config of evaluator.
   :type config: Config


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.score', 'data.label']



   .. py:method:: used_info(dataobject)

      Get scores that model predicted and the ground truth.



   .. py:method:: output_metric(metric, dataobject)


   .. py:method:: metric_info(preds, trues)
      :abstractmethod:


      Calculate the value of the metric.

      :param preds: the scores predicted by model, a one-dimensional vector.
      :type preds: numpy.ndarray
      :param trues: the label of items, which has the same shape as ``preds``.
      :type trues: numpy.ndarray

      :returns: The value of the metric.
      :rtype: float



.. py:class:: ConsumerTopKMetric(config)

   Bases: :py:obj:`AbstractMetric`


   :class:`ConsumerTopKMetric` is a base object of consumer-based metrics. If you want to
   implement a consumer-based metric, you can inherit this class.
   The consumer-based metrics are based on a binary partition of users and on the demographic parity notion,
   commonly measured as the absolute difference between the two groups in terms of a ranking metric.

   :param config: The config of evaluator.
   :type config: Config


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['eval_data.user_feat', 'rec.users']



   .. py:attribute:: smaller
      :value: True



   .. py:attribute:: USER_GROUP_1
      :value: 1



   .. py:attribute:: USER_GROUP_2
      :value: 2



   .. py:attribute:: _ranking_metric
      :value: None



   .. py:attribute:: sensitive_attribute


   .. py:property:: ranking_metric


   .. py:method:: get_group_mask(user_feat, interaction_users)


   .. py:method:: used_info(dataobject)

      Get the users features and the users in the interaction batch.



   .. py:method:: get_dp(result, group1_mask, group2_mask)

      Get the absolute difference between the two groups in terms of a ranking metric.

      :param group1_result: the result of the first group.
      :type group1_result: torch.Tensor
      :param group2_result: the result of the second group.
      :type group2_result: torch.Tensor

      :returns: the difference between the two groups in terms of a ranking metric.
      :rtype: torch.Tensor



   .. py:method:: ranking_metric_info(pos_index, pos_len)
      :abstractmethod:



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



.. py:class:: PathQualityMetric(config)

   Bases: :py:obj:`TopkMetric`


   :class:`PathQualityMetric` is a base object of path-based metrics. If you want to
   implement a path based metric, you can inherit this class.

   :param config: The config of evaluator.
   :type config: Config


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.paths']



   .. py:method:: used_info(dataobject)

      Get the bool matrix indicating whether the corresponding item is positive
      and number of positive items for each user.



   .. py:method:: normalized_ema(values)


