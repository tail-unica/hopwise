hopwise.model.exlib_recommender.lightgbm
========================================

.. py:module:: hopwise.model.exlib_recommender.lightgbm

.. autoapi-nested-parse::

   hopwise.model.exlib_recommender.lightgbm
   ##########################################



Classes
-------

.. autoapisummary::

   hopwise.model.exlib_recommender.lightgbm.LightGBM


Module Contents
---------------

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



