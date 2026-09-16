hopwise.evaluator.collector
===========================

.. py:module:: hopwise.evaluator.collector

.. autoapi-nested-parse::

   hopwise.evaluator.collector
   ################################################



Classes
-------

.. autoapisummary::

   hopwise.evaluator.collector.DataStruct
   hopwise.evaluator.collector.Collector
   hopwise.evaluator.collector.Collector_KG
   hopwise.evaluator.collector.ExplainableCollector


Module Contents
---------------

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



