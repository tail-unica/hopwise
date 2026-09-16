hopwise.evaluator.register
==========================

.. py:module:: hopwise.evaluator.register

.. autoapi-nested-parse::

   hopwise.evaluator.register
   ################################################



Attributes
----------

.. autoapisummary::

   hopwise.evaluator.register.metric_module_name


Classes
-------

.. autoapisummary::

   hopwise.evaluator.register.Register
   hopwise.evaluator.register.Register_KG


Functions
---------

.. autoapisummary::

   hopwise.evaluator.register.cluster_info


Module Contents
---------------

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


