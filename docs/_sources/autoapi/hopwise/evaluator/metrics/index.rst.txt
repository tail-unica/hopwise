hopwise.evaluator.metrics
=========================

.. py:module:: hopwise.evaluator.metrics

.. autoapi-nested-parse::

   hopwise.evaluator.metrics
   ############################

   Suppose there is a set of :math:`n` items to be ranked. Given a user :math:`u` in the user set :math:`U`,
   we use :math:`\hat R(u)` to represent a ranked list of items that a model produces, and :math:`R(u)` to
   represent a ground-truth set of items that user :math:`u` has interacted with. For top-k recommendation, only
   top-ranked items are important to consider. Therefore, in top-k evaluation scenarios, we truncate the
   recommendation list with a length :math:`K`. Besides, in loss-based metrics, :math:`S` represents the
   set of user(u)-item(i) pairs, :math:`\hat r_{u i}` represents the score predicted by the model,
   :math:`{r}_{u i}` represents the ground-truth labels.



Attributes
----------

.. autoapisummary::

   hopwise.evaluator.metrics.DeltaHit
   hopwise.evaluator.metrics.DeltaMRR
   hopwise.evaluator.metrics.DeltaMAP
   hopwise.evaluator.metrics.DeltaNDCG
   hopwise.evaluator.metrics.DeltaPrecision
   hopwise.evaluator.metrics.DeltaRecall


Classes
-------

.. autoapisummary::

   hopwise.evaluator.metrics.Hit
   hopwise.evaluator.metrics.MRR
   hopwise.evaluator.metrics.MAP
   hopwise.evaluator.metrics.Recall
   hopwise.evaluator.metrics.NDCG
   hopwise.evaluator.metrics.Precision
   hopwise.evaluator.metrics.GAUC
   hopwise.evaluator.metrics.AUC
   hopwise.evaluator.metrics.MAE
   hopwise.evaluator.metrics.RMSE
   hopwise.evaluator.metrics.LogLoss
   hopwise.evaluator.metrics.ItemCoverage
   hopwise.evaluator.metrics.AveragePopularity
   hopwise.evaluator.metrics.ShannonEntropy
   hopwise.evaluator.metrics.GiniIndex
   hopwise.evaluator.metrics.TailPercentage
   hopwise.evaluator.metrics.Serendipity
   hopwise.evaluator.metrics.Novelty
   hopwise.evaluator.metrics.LIR
   hopwise.evaluator.metrics.Fidelity
   hopwise.evaluator.metrics.SEP
   hopwise.evaluator.metrics.LID
   hopwise.evaluator.metrics.SED
   hopwise.evaluator.metrics.PTD
   hopwise.evaluator.metrics.PTC
   hopwise.evaluator.metrics.PPT
   hopwise.evaluator.metrics.LITD
   hopwise.evaluator.metrics.SETD


Functions
---------

.. autoapisummary::

   hopwise.evaluator.metrics.create_consumer_metric_class


Module Contents
---------------

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



