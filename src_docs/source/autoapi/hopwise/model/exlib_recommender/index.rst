hopwise.model.exlib_recommender
===============================

.. py:module:: hopwise.model.exlib_recommender


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/model/exlib_recommender/lightgbm/index
   /autoapi/hopwise/model/exlib_recommender/xgboost/index


Classes
-------

.. autoapisummary::

   hopwise.model.exlib_recommender.LightGBM
   hopwise.model.exlib_recommender.XGBoost


Package Contents
----------------

.. py:class:: LightGBM(config, dataset)

   Bases: :py:obj:`lightgbm.Booster`


   LightGBM is inherited from lgb.Booster


   .. py:attribute:: type


   .. py:attribute:: input_type


   .. py:method:: to(device)


   .. py:method:: load_state_dict(model_file)

      Load state dictionary

      :param model_file: file path of saved model
      :type model_file: str



   .. py:method:: load_other_parameter(other_parameter)

      Load other parameters



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



