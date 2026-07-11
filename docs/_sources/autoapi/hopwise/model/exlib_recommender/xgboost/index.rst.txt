hopwise.model.exlib_recommender.xgboost
=======================================

.. py:module:: hopwise.model.exlib_recommender.xgboost

.. autoapi-nested-parse::

   hopwise.model.exlib_recommender.xgboost
   ########################################



Classes
-------

.. autoapisummary::

   hopwise.model.exlib_recommender.xgboost.XGBoost


Module Contents
---------------

.. py:class:: XGBoost(config, dataset)

   Bases: :py:obj:`xgboost.Booster`


   XGBoost is inherited from xgb.Booster


   .. py:attribute:: type


   .. py:attribute:: input_type


   .. py:method:: to(device)


   .. py:method:: load_state_dict(model_file)

      Load state dictionary

      :param model_file: file path of saved model
      :type model_file: str



   .. py:method:: load_other_parameter(other_parameter)

      Load other parameters



