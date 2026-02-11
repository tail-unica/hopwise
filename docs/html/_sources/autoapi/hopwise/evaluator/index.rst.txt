hopwise.evaluator
=================

.. py:module:: hopwise.evaluator


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/evaluator/base_metric/index
   /autoapi/hopwise/evaluator/collector/index
   /autoapi/hopwise/evaluator/evaluator/index
   /autoapi/hopwise/evaluator/metrics/index
   /autoapi/hopwise/evaluator/register/index
   /autoapi/hopwise/evaluator/utils/index


Attributes
----------

.. autoapisummary::

   hopwise.evaluator.DeltaHit
   hopwise.evaluator.DeltaMRR
   hopwise.evaluator.DeltaMAP
   hopwise.evaluator.DeltaNDCG
   hopwise.evaluator.DeltaPrecision
   hopwise.evaluator.DeltaRecall
   hopwise.evaluator.metric_module_name


Classes
-------

.. autoapisummary::

   hopwise.evaluator.EvaluatorType
   hopwise.evaluator.AbstractMetric
   hopwise.evaluator.TopkMetric
   hopwise.evaluator.LossMetric
   hopwise.evaluator.ConsumerTopKMetric
   hopwise.evaluator.PathQualityMetric
   hopwise.evaluator.AbstractMetric
   hopwise.evaluator.ConsumerTopKMetric
   hopwise.evaluator.LossMetric
   hopwise.evaluator.PathQualityMetric
   hopwise.evaluator.TopkMetric
   hopwise.evaluator.EvaluatorType
   hopwise.evaluator.Hit
   hopwise.evaluator.MRR
   hopwise.evaluator.MAP
   hopwise.evaluator.Recall
   hopwise.evaluator.NDCG
   hopwise.evaluator.Precision
   hopwise.evaluator.GAUC
   hopwise.evaluator.AUC
   hopwise.evaluator.MAE
   hopwise.evaluator.RMSE
   hopwise.evaluator.LogLoss
   hopwise.evaluator.ItemCoverage
   hopwise.evaluator.AveragePopularity
   hopwise.evaluator.ShannonEntropy
   hopwise.evaluator.GiniIndex
   hopwise.evaluator.TailPercentage
   hopwise.evaluator.Serendipity
   hopwise.evaluator.Novelty
   hopwise.evaluator.LIR
   hopwise.evaluator.Fidelity
   hopwise.evaluator.SEP
   hopwise.evaluator.LID
   hopwise.evaluator.SED
   hopwise.evaluator.PTD
   hopwise.evaluator.PTC
   hopwise.evaluator.PPT
   hopwise.evaluator.LITD
   hopwise.evaluator.SETD
   hopwise.evaluator.DataStruct
   hopwise.evaluator.Evaluator
   hopwise.evaluator.Evaluator_KG
   hopwise.evaluator.Register
   hopwise.evaluator.Register_KG
   hopwise.evaluator.Register
   hopwise.evaluator.Register_KG
   hopwise.evaluator.DataStruct
   hopwise.evaluator.Collector
   hopwise.evaluator.Collector_KG
   hopwise.evaluator.ExplainableCollector


Functions
---------

.. autoapisummary::

   hopwise.evaluator._binary_clf_curve
   hopwise.evaluator.create_consumer_metric_class
   hopwise.evaluator.cluster_info
   hopwise.evaluator.train_tsne


Package Contents
----------------

.. py:class:: EvaluatorType

   Bases: :py:obj:`enum.Enum`


   Type for evaluation metrics.

   - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
   - ``VALUE``: Value-based metrics like AUC, etc.


   .. py:attribute:: RANKING
      :value: 1



   .. py:attribute:: VALUE
      :value: 2



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



.. py:function:: _binary_clf_curve(trues, preds)

   Calculate true and false positives per binary classification threshold

   :param trues: the true scores' list
   :type trues: numpy.ndarray
   :param preds: the predict scores' list
   :type preds: numpy.ndarray

   :returns: A count of false positives, at index i being the number of negative
             samples assigned a score >= thresholds[i]
             preds (numpy.ndarray): An increasing count of true positives, at index i being the number
             of positive samples assigned a score >= thresholds[i].
   :rtype: fps (numpy.ndarray)

   .. note::

      To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
      in SkLearn and made some optimizations.


.. py:class:: EvaluatorType

   Bases: :py:obj:`enum.Enum`


   Type for evaluation metrics.

   - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
   - ``VALUE``: Value-based metrics like AUC, etc.


   .. py:attribute:: RANKING
      :value: 1



   .. py:attribute:: VALUE
      :value: 2



.. py:class:: Hit(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
   you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
   we call it a hit.

   .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

   .. math::
       \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

   :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
   :math:`\emptyset` denotes the empty set.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: MRR(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
   of the first relevant item found by an algorithm.

   .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

   .. math::
      \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

   :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: MAP(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   MAP_ (also known as Mean Average Precision) is meant to calculate
   average precision for the relevant items.

   .. note::

      In this case the normalization factor used is :math:`\frac{1}{min(|\hat R(u)|, K)}`, which prevents your
      AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture
      all the correct ones.

   .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

   .. math::
      \mathrm{MAP@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{min(|\hat R(u)|, K)} \sum_{j=1}^{|\hat{R}(u)|} I\left(\hat{R}_{j}(u) \in R(u)\right) \cdot  Precision@j)

   :math:`\hat{R}_{j}(u)` is the j-th item in the recommendation list of \hat R (u)).


   .. py:attribute:: config


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index, pos_len)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: Recall(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   Recall_ is a measure for computing the fraction of relevant items out of all relevant items.

   .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

   .. math::
      \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

   :math:`|R(u)|` represents the item count of :math:`R(u)`.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index, pos_len)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: NDCG(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
   where positions are discounted logarithmically. It accounts for the position of the hit by assigning
   higher scores to hits at top ranks.

   .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

   .. math::
       \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
       \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

   :math:`\delta(·)` is an indicator function.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index, pos_len)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: Precision(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.TopkMetric`


   Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
   out of all the recommended items. We average the metric for each user :math:`u` get the final result.

   .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

   .. math::
       \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}

   :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_index)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



.. py:class:: GAUC(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   GAUC (also known as Grouped Area Under Curve) is used to evaluate the two-class model, referring to
   the area under the ROC curve grouped by user. We weighted the index of each user :math:`u` by the number of positive
   samples of users to get the final result.

   For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3219819.3219823>`__

   .. note::

      It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
      It is also not limited to k. Due to our padding for `scores_tensor` with `-np.inf`, the padding
      value will influence the ranks of origin items. Therefore, we use descending sort here and make
      an identity transformation  to the formula of `AUC`, which is shown in `auc_` function.
      For readability, we didn't do simplification in the code.

   .. math::
       \begin{align*}
           \mathrm {AUC(u)} &= \frac {{{|R(u)|} \times {(n+1)} - \frac{|R(u)| \times (|R(u)|+1)}{2}} -
           \sum\limits_{i=1}^{|R(u)|} rank_{i}} {{|R(u)|} \times {(n - |R(u)|)}} \\
           \mathrm{GAUC} &= \frac{1}{\sum_{u \in U} |R(u)|}\sum_{u \in U} |R(u)| \cdot(\mathrm {AUC(u)})
       \end{align*}

   :math:`rank_i` is the descending rank of the i-th items in :math:`R(u)`.


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.meanrank']



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(pos_rank_sum, user_len_list, pos_len_list)

      Get the value of GAUC metric.

      :param pos_rank_sum: sum of descending rankings for positive items of each users.
      :type pos_rank_sum: numpy.ndarray
      :param user_len_list: the number of predicted items for users.
      :type user_len_list: numpy.ndarray
      :param pos_len_list: the number of positive items for users.
      :type pos_len_list: numpy.ndarray

      :returns: The value of the GAUC.
      :rtype: float



.. py:class:: AUC(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.LossMetric`


   AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
   the area under the ROC curve.

   .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

   .. note::

      This metric does not calculate group-based AUC which considers the AUC scores
      averaged across users. It is also not limited to k. Instead, it calculates the
      scores on the entire prediction results regardless the users. We call the interface
      in `scikit-learn`, and code calculates the metric using the variation of following formula.

   .. math::
       \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
       \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

   :math:`M` denotes the number of positive items.
   :math:`N` denotes the total number of user-item interactions.
   :math:`rank_i` denotes the descending rank of the i-th positive item.


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(preds, trues)

      Calculate the value of the metric.

      :param preds: the scores predicted by model, a one-dimensional vector.
      :type preds: numpy.ndarray
      :param trues: the label of items, which has the same shape as ``preds``.
      :type trues: numpy.ndarray

      :returns: The value of the metric.
      :rtype: float



.. py:class:: MAE(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.LossMetric`


   MAE_ (also known as Mean Absolute Error regression loss) is used to evaluate the difference between
   the score predicted by the model and the actual behavior of the user.

   .. _MAE: https://en.wikipedia.org/wiki/Mean_absolute_error

   .. math::
       \mathrm{MAE}=\frac{1}{|{S}|} \sum_{(u, i) \in {S}}\left|\hat{r}_{u i}-r_{u i}\right|

   :math:`|S|` represents the number of pairs in :math:`S`.


   .. py:attribute:: smaller
      :value: True



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(preds, trues)

      Calculate the value of the metric.

      :param preds: the scores predicted by model, a one-dimensional vector.
      :type preds: numpy.ndarray
      :param trues: the label of items, which has the same shape as ``preds``.
      :type trues: numpy.ndarray

      :returns: The value of the metric.
      :rtype: float



.. py:class:: RMSE(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.LossMetric`


   RMSE_ (also known as Root Mean Squared Error) is another error metric like `MAE`.

   .. _RMSE: https://en.wikipedia.org/wiki/Root-mean-square_deviation

   .. math::
      \mathrm{RMSE} = \sqrt{\frac{1}{|{S}|} \sum_{(u, i) \in {S}}(\hat{r}_{u i}-r_{u i})^{2}}


   .. py:attribute:: smaller
      :value: True



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(preds, trues)

      Calculate the value of the metric.

      :param preds: the scores predicted by model, a one-dimensional vector.
      :type preds: numpy.ndarray
      :param trues: the label of items, which has the same shape as ``preds``.
      :type trues: numpy.ndarray

      :returns: The value of the metric.
      :rtype: float



.. py:class:: LogLoss(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.LossMetric`


   Logloss_ (also known as logistic loss or cross-entropy loss) is used to evaluate the probabilistic
   output of the two-class classifier.

   .. _Logloss: http://wiki.fast.ai/index.php/Log_Loss

   .. math::
       LogLoss = \frac{1}{|S|} \sum_{(u,i) \in S}(-((r_{u i} \ \log{\hat{r}_{u i}}) + {(1 - r_{u i})}\ \log{(1 - \hat{r}_{u i})}))


   .. py:attribute:: smaller
      :value: True



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(preds, trues)

      Calculate the value of the metric.

      :param preds: the scores predicted by model, a one-dimensional vector.
      :type preds: numpy.ndarray
      :param trues: the label of items, which has the same shape as ``preds``.
      :type trues: numpy.ndarray

      :returns: The value of the metric.
      :rtype: float



.. py:class:: ItemCoverage(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   ItemCoverage_ computes the coverage of recommended items over all items.

   .. _ItemCoverage: https://en.wikipedia.org/wiki/Coverage_(information_systems)

   For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/1864708.1864761>`__
   and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

   .. math::
      \mathrm{Coverage@K}=\frac{\left| \bigcup_{u \in U} \hat{R}(u) \right|}{|I|}


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.num_items']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and number of items in total item set



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_coverage(item_matrix, num_items)

      Get the coverage of recommended items over all items

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray
      :param num_items: the total number of items.
      :type num_items: int

      :returns: the `coverage` metric.
      :rtype: float



.. py:class:: AveragePopularity(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   AveragePopularity computes the average popularity of recommended items.

   For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
   and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

   .. math::
       \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

   :math:`\phi(i)` is the number of interaction of item i in training data.


   .. py:attribute:: metric_type


   .. py:attribute:: smaller
      :value: True



   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.count_items']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and the popularity of items in training data



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_pop(item_matrix, item_count)

      Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray
      :param item_count: the number of interaction of items in training data.
      :type item_count: dict

      :returns: the popularity of items in the recommended list.
      :rtype: numpy.ndarray



   .. py:method:: metric_info(values)


   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: ShannonEntropy(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   ShannonEntropy_ presents the diversity of the recommendation items.
   It is the entropy over items' distribution.

   .. _ShannonEntropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

   For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
   and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__

   .. math::
       \mathrm {ShannonEntropy@K}=-\sum_{i=1}^{|I|} p(i) \log p(i)

   :math:`p(i)` is the probability of recommending item i
   which is the number of item i in recommended list over all items.


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.items']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items.



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_entropy(item_matrix)

      Get shannon entropy through the top-k recommendation list.

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray

      :returns: the shannon entropy.
      :rtype: float



.. py:class:: GiniIndex(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   GiniIndex presents the diversity of the recommendation items.
   It is used to measure the inequality of a distribution.

   .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient

   For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3308560.3317303>`__.

   .. math::
       \mathrm {GiniIndex@K}=\left(\frac{\sum_{i=1}^{|I|}(2 i-|I|-1) P{(i)}}{|I| \sum_{i=1}^{|I|} P{(i)}}\right)

   :math:`P{(i)}` represents the number of times all items appearing in the recommended list,
   which is indexed in non-decreasing order (P_{(i)} \leq P_{(i+1)}).


   .. py:attribute:: metric_type


   .. py:attribute:: smaller
      :value: True



   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.num_items']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and number of items in total item set



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_gini(item_matrix, num_items)

      Get gini index through the top-k recommendation list.

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray
      :param num_items: the total number of items.
      :type num_items: int

      :returns: the gini index.
      :rtype: float



.. py:class:: TailPercentage(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   TailPercentage_ computes the percentage of long-tail items in recommendation items.

   .. _TailPercentage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

   For further details, please refer to the `paper <https://arxiv.org/pdf/2007.12329.pdf>`__.

   .. math::
       \mathrm {TailPercentage@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_{u}} {\delta(i \in T)}}{|R_{u}|}

   :math:`\delta(·)` is an indicator function.
   :math:`T` is the set of long-tail items,
   which is a portion of items that appear in training data seldomly.

   .. note::

      If you want to use this metric, please set the parameter 'tail_ratio' in the config
      which can be an integer or a float in (0,1]. Otherwise it will default to 0.1.


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.count_items']



   .. py:attribute:: topk


   .. py:attribute:: tail


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and number of items in total item set.



   .. py:method:: get_tail(item_matrix, count_items)

      Get long-tail percentage through the top-k recommendation list.

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray
      :param count_items: the number of interaction of items in training data.
      :type count_items: dict

      :returns: long-tail percentage.
      :rtype: float



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(values)


   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:function:: create_consumer_metric_class(topk_metric)

   Dynamically creates and returns a new consumer class with the given name, base classes, and attributes.


.. py:data:: DeltaHit

.. py:data:: DeltaMRR

.. py:data:: DeltaMAP

.. py:data:: DeltaNDCG

.. py:data:: DeltaPrecision

.. py:data:: DeltaRecall

.. py:class:: Serendipity(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   Serendipity is a measure of how surprising the recommended items are to the user.
   It is defined as the fraction of recommended items that are not popular in the training data.


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.num_items', 'data.num_users', 'data.count_items', 'data.history_index']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and the popularity of items in training data



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(values)


   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: Novelty(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.AbstractMetric`


   Paper:
   Novelty: Inverse of popularity of the items recommended to the user


   .. py:attribute:: metric_type


   .. py:attribute:: metric_need
      :value: ['rec.items', 'data.count_items', 'data.num_items']



   .. py:attribute:: topk


   .. py:method:: used_info(dataobject)

      Get the matrix of recommendation items and number of items in total item set



   .. py:method:: get_pop(item_matrix, item_count)

      Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

      :param item_matrix: matrix of items recommended to users.
      :type item_matrix: numpy.ndarray
      :param item_count: the number of interaction of items in training data.
      :type item_count: dict

      :returns: the popularity of items in the recommended list.
      :rtype: numpy.ndarray



   .. py:method:: normalize_popularity(item_matrix, pop_matrix, item_count, num_items)


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



.. py:class:: LIR(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Linking Interaction Recency (LIR)

   This property serves to quantify the time since the linking interaction in the explanation path occurred.
   Given a user :math:`u \in U` and the set :math:`P_u` of items this user interacted with,
   we denote the list of their interactions, sorted chronologically, by :math:`T_u = [(p^i, t^i)]`,
   where :math:`p^i \in P_u` is a item experienced by the user, :math:`t^i \in \mathbb N`
   is the timestamp that interaction occurred, and :math:`t^i \leq t^{i+1}` :math:`\forall i = 1, \dots, |P_u|`.

   They applied an exponentially weighed moving average to the timestamps included in :math:`T_u`,
   to obtain the LIR of each interaction performed by the user :math:`u`.
   Specifically, given an interaction :math:`(p^i, t^i) \in T_u`, the LIR for that interaction
   was computed as follows:

   .. math::
       \mathrm{LIR(p^i, t^i)} = ( 1 - \beta_{LIR} ) \cdot LIR(p^{i-1}, t^{i-1}) + \beta_{LIR} \cdot t^i


   where :math:`\beta_{LIR} \in [0, 1]` is a decay associated to the interaction time,
   and :math:`LIR(p^1, t^1) = t^1`.
   The LIR values were min-max normalized for each user to lay in the range :math:`[0, 1]`,
   with values close to 0 (1) meaning that the linking interaction is far away (recent) in time.
   In the case of a recommended item not being present in the user's interaction history,
   the LIR of the linking interaction was set to 0, meaning that the linking interaction
   is very recent (i.e., the user has just interacted with the item).
   The overall LIR for explanations in a recommended list was obtained
   by averaging the LIR of the linking interactions for the selected explanation path of each recommended item.

   For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3477495.3532041>`.


   .. py:attribute:: metric_need
      :value: ['data.timestamp', 'data.num_items']



   .. py:attribute:: li_idx
      :value: 1



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_lir_matrix(timestamp_matrix)


   .. py:method:: metric_info(lir_matrix, paths, num_items)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: Fidelity(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Fidelity


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to
      :type value: numpy.ndarray
      :param `metric@max:
      :type `metric@max: self.topk

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: SEP(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Popularity of Shared Entity (SEP)

   This property serves to quantify the extent to which the shared entity included in an explanation-path is popular.
   They assume that the number of relationships a shared entity is involved in the KG is a proxy of its popularity.
   For instance, the popularity of an actor is computed by counting how many movies that actor starred in.
   They denote the list of entities of a given type $\lambda$ in the KG, sorted based on their popularity,
   by :math:`_\lambda = [(e^i, v^i)]`, where :math:`e^i \in E_\lambda` is an entity
   of type :math:`\lambda`, :math:`v^i \in \mathbb N` is the number of relationships a shared entity is involved in
   (in-degree),
   and :math:`v^i \leq v^{i+1}` :math:`\forall i = 1, \dots, |E_\lambda|`.
   We applied an exponential decay to the popularity scores in :math:`S_\lambda$, to get the SEP of an entity
   of type :math:`\lambda`, as:

   .. math::
       \mathrm{SEP(e^i, v^i)} = ( 1 - \beta_{SEP} ) \cdot SEP(e^{i-1}, v^{i-1}) + \beta_{SEP} \cdot v^i

   where :math:`\beta_{SEP}` is a decay related to the popularity, and :math:`SEP(e^1, v^1) = v^1`
   The SEP values w ere min-max normalized for each entity type to lay in the range [0, 1],
   with values close to 0 (1) when the entity has a low (high) popularity.
   The shared entity novelty might be obtained as :math:`SEN(e^i, v^i) = 1-SEP(e^i, v^i).

   The overall SEP for explanations in a recommended list was obtained by averaging the SEP
   values of the shared entity # noqa: E501
   in the selected path for each recommended item.

   For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3477495.3532041>`.


   .. py:attribute:: metric_need
      :value: ['data.node_degree']



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: get_sep_matrix(node_degree)


   .. py:method:: metric_info(paths, sep_matrix)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: LID(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Diversity of Linked Interaction (LID)



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: SED(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Diversity of Shared Entities



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: PTD(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Path Type Diversity


   .. py:attribute:: metric_need
      :value: ['data.max_path_type']



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths, max_path_type)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: PTC(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Path Type Concentration (PTC)


   .. py:attribute:: metric_need
      :value: ['data.max_path_type']



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths, max_path_type)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: PPT(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Path Pattern Type


   .. py:attribute:: metric_need
      :value: ['data.max_path_length', 'data.rid2relation']



   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths, max_path_pattern, rid2r_name)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: LITD(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Linked Interaction Type Diversity


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: SETD(config)

   Bases: :py:obj:`hopwise.evaluator.base_metric.PathQualityMetric`


   Shared Entities Type Diversity


   .. py:method:: calculate_metric(dataobject)

      Get the dictionary of a metric.

      :param dataobject: it contains all the information needed to calculate metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
      :rtype: dict



   .. py:method:: metric_info(paths)

      Calculate the value of the metric.

      :param pos_index: a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th             highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
      :type pos_index: numpy.ndarray
      :param pos_len: a vector representing the number of positive items per user, shape of ``(n_users,)``.
      :type pos_len: numpy.ndarray

      :returns: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :rtype: numpy.ndarray



   .. py:method:: topk_result(metric, value)

      Match the metric value to the `k` and put them in `dictionary` form.

      :param metric: the name of calculated metric.
      :type metric: str
      :param value: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
      :type value: numpy.ndarray

      :returns: metric values required in the configuration.
      :rtype: dict



.. py:class:: DataStruct

   .. py:attribute:: _data_dict


   .. py:method:: __getitem__(name: str)


   .. py:method:: __setitem__(name: str, value)


   .. py:method:: __delitem__(name: str)


   .. py:method:: __contains__(key: str)


   .. py:method:: get(name: str)


   .. py:method:: set(name: str, value)


   .. py:method:: update_tensor(name: str, value)


   .. py:method:: __str__()


.. py:class:: Evaluator(config)

   Evaluator is used to check parameter correctness, and summarize the results of all metrics.


   .. py:attribute:: config


   .. py:attribute:: metrics


   .. py:attribute:: metric_class


   .. py:method:: evaluate(dataobject: hopwise.evaluator.collector.DataStruct)

      Calculate all the metrics. It is called at the end of each epoch

      :param dataobject: It contains all the information needed for metrics.
      :type dataobject: DataStruct

      :returns: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``
      :rtype: collections.OrderedDict



.. py:class:: Evaluator_KG(config)

   Bases: :py:obj:`Evaluator`


   Evaluator KG extends the Evaluator class for link prediction tasks.


   .. py:attribute:: metrics


   .. py:attribute:: metric_class


.. py:function:: cluster_info(module_name)

   Collect information of all metrics, including:

       - ``metric_need``: Information needed to calculate this metric, the combination of ``rec.items, rec.topk,
         rec.meanrank, rec.score, data.num_items, data.num_users, data.count_items, data.count_users, data.label``.
       - ``metric_type``: Whether the scores required by metric are grouped by user, range in ``EvaluatorType.RANKING``
         and ``EvaluatorType.VALUE``.
       - ``smaller``: Whether the smaller metric value represents better performance,
         range in ``True`` and ``False``, default to ``False``.

   .. note::

      For ``metric_type``: in current hopwise, all the "grouped-score" metrics are ranking-based and all the
      "non-grouped-score" metrics are value-based. To keep with our paper, we adopted the more formal terms:
      ``RANKING`` and ``VALUE``.

   :param module_name: the name of module ``hopwise.evaluator.metrics``.
   :type module_name: str

   :returns: Three dictionaries containing the above information
             and a dictionary matching metric names to metric classes.
   :rtype: dict


.. py:data:: metric_module_name
   :value: 'hopwise.evaluator.metrics'


.. py:class:: Register(config)

   Register module load the registry according to the metrics in config.
   It is a member of DataCollector.
   The DataCollector collect the resource that need for Evaluator under the guidance of Register


   .. py:attribute:: config


   .. py:attribute:: metrics


   .. py:method:: _build_register()


   .. py:method:: has_metric(metric: str)


   .. py:method:: need(key: str)


.. py:class:: Register_KG(config)

   Bases: :py:obj:`Register`


   Register module load the registry according to the metrics in config.
   It is a member of DataCollector.
   The DataCollector collect the resource that need for Evaluator under the guidance of Register


   .. py:attribute:: metrics


.. py:class:: Register(config)

   Register module load the registry according to the metrics in config.
   It is a member of DataCollector.
   The DataCollector collect the resource that need for Evaluator under the guidance of Register


   .. py:attribute:: config


   .. py:attribute:: metrics


   .. py:method:: _build_register()


   .. py:method:: has_metric(metric: str)


   .. py:method:: need(key: str)


.. py:class:: Register_KG(config)

   Bases: :py:obj:`Register`


   Register module load the registry according to the metrics in config.
   It is a member of DataCollector.
   The DataCollector collect the resource that need for Evaluator under the guidance of Register


   .. py:attribute:: metrics


.. py:function:: train_tsne(model, config, load_best_model)

.. py:class:: DataStruct

   .. py:attribute:: _data_dict


   .. py:method:: __getitem__(name: str)


   .. py:method:: __setitem__(name: str, value)


   .. py:method:: __delitem__(name: str)


   .. py:method:: __contains__(key: str)


   .. py:method:: get(name: str)


   .. py:method:: set(name: str, value)


   .. py:method:: update_tensor(name: str, value)


   .. py:method:: __str__()


.. py:class:: Collector(config)

   The collector is used to collect the resource for evaluator.
   As the evaluation metrics are various, the needed resource not only contain the recommended result
   but also other resource from data and model. They all can be collected by the collector during the training
   and evaluation process.

   This class is only used in Trainer.



   .. py:attribute:: config


   .. py:attribute:: data_struct


   .. py:attribute:: register


   .. py:attribute:: full


   .. py:attribute:: topk


   .. py:attribute:: device


   .. py:method:: train_data_collect(train_data)

      Collect the evaluation resource from training data.

      :param train_data: the training dataloader which contains the training data.
      :type train_data: AbstractDataLoader



   .. py:method:: eval_data_collect(eval_data)

      Collect the evaluation resource from evaluation data, such as user and item features.

      :param eval_data: the evaluation dataloader which contains the evaluation data.
      :type eval_data: AbstractDataLoader



   .. py:method:: _average_rank(scores)

      Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

      :param scores: an ordered tensor, with size of `(N, )`
      :type scores: tensor

      :returns: average_rank
      :rtype: torch.Tensor

      .. rubric:: Example

      >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
      tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
      [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

      Reference:
          https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352




   .. py:method:: eval_batch_collect(scores, interaction, positive_u: torch.Tensor, positive_i: torch.Tensor)

      Collect the evaluation resource from batched eval data and batched model output.

      :param scores: the output tensor of model with the shape of `(N, )`
      :type scores: Torch.Tensor
      :param interaction: batched eval data.
      :type interaction: Interaction
      :param positive_u: the row index of positive items for each user.
      :type positive_u: Torch.Tensor
      :param positive_i: the positive item id for each user.
      :type positive_i: Torch.Tensor



   .. py:method:: model_collect(model: torch.nn.Module, load_best_model=False)

      Collect the evaluation resource from model and do something with the model.

      :param model: the trained recommendation model.
      :type model: nn.Module
      :param load_best_model: whether to load the best model.
      :type load_best_model: bool



   .. py:method:: eval_collect(eval_pred: torch.Tensor, data_label: torch.Tensor)

      Collect the evaluation resource from total output and label.
      It was designed for those models that can not predict with batch.

      :param eval_pred: the output score tensor of model.
      :type eval_pred: torch.Tensor
      :param data_label: the label tensor.
      :type data_label: torch.Tensor



   .. py:method:: get_data_struct()

      Get all the evaluation resource that been collected.
      And reset some of outdated resource.



.. py:class:: Collector_KG(config)

   Bases: :py:obj:`Collector`


   This collector is used to collect the resource for evaluator in knowledge graph embedding models.
   Specifically, it collects the predictions for the link prediction task, extending Collector from recommendation.



   .. py:attribute:: register


   .. py:attribute:: topk


.. py:class:: ExplainableCollector(config)

   Bases: :py:obj:`Collector`


   This collector is used to collect the resource for evaluator in explainable recommendation models.
   It collects the KG paths and explanations for the recommendations made by the model and enables
   path quality evaluation.



   .. py:attribute:: register


   .. py:method:: train_data_collect(train_data)

      Collect the evaluation resource from training data.

      :param train_data: the training dataloader which contains the training data.
      :type train_data: AbstractDataLoader



   .. py:method:: node_degree_dict(train_data)


   .. py:method:: eval_batch_collect(explanations, interaction, positive_u: torch.Tensor, positive_i: torch.Tensor)

      Collect the evaluation resource from batched eval data and batched model output.

      :param explanations: a tuple containing the scores and paths, where:
                           - scores (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                           - paths (list): a list of quadruples representing the paths for each user.
      :type explanations: tuple
      :param interaction: batched eval data.
      :type interaction: Interaction
      :param positive_u: the row index of positive items for each user.
      :type positive_u: Torch.Tensor
      :param positive_i: the positive item id for each user.
      :type positive_i: Torch.Tensor



