hopwise.evaluator.evaluator
===========================

.. py:module:: hopwise.evaluator.evaluator

.. autoapi-nested-parse::

   hopwise.evaluator.evaluator
   #####################################



Classes
-------

.. autoapisummary::

   hopwise.evaluator.evaluator.Evaluator
   hopwise.evaluator.evaluator.Evaluator_KG


Module Contents
---------------

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


