hopwise.trainer.hyper_tuning
============================

.. py:module:: hopwise.trainer.hyper_tuning

.. autoapi-nested-parse::

   hopwise.trainer.hyper_tuning
   ############################



Exceptions
----------

.. autoapisummary::

   hopwise.trainer.hyper_tuning.ExhaustiveSearchError


Classes
-------

.. autoapisummary::

   hopwise.trainer.hyper_tuning.HyperTuning


Functions
---------

.. autoapisummary::

   hopwise.trainer.hyper_tuning._recursiveFindNodes
   hopwise.trainer.hyper_tuning._parameters
   hopwise.trainer.hyper_tuning._spacesize
   hopwise.trainer.hyper_tuning.exhaustive_search


Module Contents
---------------

.. py:function:: _recursiveFindNodes(root, node_type='switch')

.. py:function:: _parameters(space)

.. py:function:: _spacesize(space)

.. py:exception:: ExhaustiveSearchError

   Bases: :py:obj:`Exception`


   ExhaustiveSearchError


.. py:function:: exhaustive_search(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000)

   This is for exhaustive search in HyperTuning.


.. py:class:: HyperTuning(objective_function, tuner='optuna', space=None, params_file=None, params_dict=None, fixed_config_file_list=None, display_file=None, algo=None, max_evals=100, early_stop=10, output_path=None, timeout=None, show_progress=False, study_name=None, resume=False)

   HyperTuning Class is used to manage the parameter tuning process of recommender system models.
   Given objective funciton, parameters range and optimization algorithm, using HyperTuning can find
   the best result among these parameters.

   .. note::

      HyperTuning provides three tuner tools:
          - hyperopt (https://github.com/hyperopt/hyperopt)
          - ray (https://docs.ray.io/en/latest/tune/index.html)
          - optuna (https://optuna.org/)
      
      Thanks to sbrodeur for the exhaustive search code.
      https://github.com/hyperopt/hyperopt/issues/200


   .. py:attribute:: PARAMS_PER_ROW
      :value: 3



   .. py:attribute:: TUNER_TYPES


   .. py:attribute:: tuner


   .. py:attribute:: best_score
      :value: None



   .. py:attribute:: best_params
      :value: None



   .. py:attribute:: best_test_result
      :value: None



   .. py:attribute:: params2result


   .. py:attribute:: params_list
      :value: []



   .. py:attribute:: score_list
      :value: []



   .. py:attribute:: show_progress
      :value: False



   .. py:attribute:: objective_function


   .. py:attribute:: max_evals
      :value: 100



   .. py:attribute:: timeout
      :value: None



   .. py:attribute:: fixed_config_file_list
      :value: None



   .. py:attribute:: display_file
      :value: None



   .. py:attribute:: output_path
      :value: '.'



   .. py:attribute:: study_name
      :value: 'hyper_Uninferable'



   .. py:attribute:: resume
      :value: False



   .. py:method:: select_algo(algo)

      Select the algorithm for hyperparameter tuning
      :param algo: the algorithm name or function
      :type algo: str or callable



   .. py:method:: select_early_stop(early_stop_steps)


   .. py:method:: _get_tuner_distributions()


   .. py:method:: build_space_from_file(file)


   .. py:method:: build_space_from_dict(config_dict)


   .. py:method:: _build_space_from_file(file, choice_fn=None, uniform_fn=None, quniform_fn=None, loguniform_fn=None)
      :staticmethod:



   .. py:method:: _build_space_from_dict(config_dict, choice_fn=None, uniform_fn=None, quniform_fn=None, loguniform_fn=None)
      :staticmethod:



   .. py:method:: build_optuna_space(trial)

      Build the space for optuna

      :param trial: the trial object
      :type trial: optuna.trial



   .. py:method:: params2str(params)
      :staticmethod:


      Convert dict to str

      :param params: parameters dict
      :type params: dict

      :returns: parameters string
      :rtype: str



   .. py:method:: _print_result(result_dict: dict)
      :staticmethod:



   .. py:method:: export_result(output_path=None)

      Write the searched parameters and corresponding results to the file

      :param output_path: the output file
      :type output_path: str



   .. py:method:: trial(params)

      Given a set of parameters, return results and optimization status

      :param params: the parameter dictionary
      :type params: dict



   .. py:method:: plot_hyper()


   .. py:method:: run()

      Begin to search the best parameters



