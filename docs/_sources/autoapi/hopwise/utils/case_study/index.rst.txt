hopwise.utils.case_study
========================

.. py:module:: hopwise.utils.case_study

.. autoapi-nested-parse::

   hopwise.utils.case_study
   #####################################



Functions
---------

.. autoapisummary::

   hopwise.utils.case_study.full_sort_scores
   hopwise.utils.case_study.full_sort_explanations
   hopwise.utils.case_study.full_sort_topk


Module Contents
---------------

.. py:function:: full_sort_scores(uid_series, model, test_data, device=None)

   Calculate the scores of all items for each user in uid_series.

   .. note:: The score of [pad] and history items will be set into -inf.

   :param uid_series: User id series.
   :type uid_series: numpy.ndarray or list
   :param model: Model to predict.
   :type model: AbstractRecommender
   :param test_data: The test_data of model.
   :type test_data: FullSortEvalDataLoader
   :param device: The device which model will run on. Defaults to ``None``.
                  Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.
   :type device: torch.device, optional

   :returns: the scores of all items for each user in uid_series.
   :rtype: torch.Tensor


.. py:function:: full_sort_explanations(uid_series, model, test_data, device=None)

   Calculate the scores of all items for each user in uid_series.

   .. note:: The score of [pad] and history items will be set into -inf.

   :param uid_series: User id series.
   :type uid_series: numpy.ndarray or list
   :param model: Model to predict.
   :type model: AbstractRecommender
   :param test_data: The test_data of model.
   :type test_data: FullSortEvalDataLoader
   :param device: The device which model will run on. Defaults to ``None``.
                  Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.
   :type device: torch.device, optional

   :returns: the scores of all items for each user in uid_series.
   :rtype: torch.Tensor


.. py:function:: full_sort_topk(uid_series, model, test_data, k, device=None)

   Calculate the top-k items' scores and ids for each user in uid_series.

   .. note:: The score of [pad] and history items will be set into -inf.

   :param uid_series: User id series.
   :type uid_series: numpy.ndarray
   :param model: Model to predict.
   :type model: AbstractRecommender
   :param test_data: The test_data of model.
   :type test_data: FullSortEvalDataLoader
   :param k: The top-k items.
   :type k: int
   :param device: The device which model will run on. Defaults to ``None``.
                  Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.
   :type device: torch.device, optional

   :returns:     - topk_scores (torch.Tensor): The scores of topk items.
                 - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
   :rtype: tuple


