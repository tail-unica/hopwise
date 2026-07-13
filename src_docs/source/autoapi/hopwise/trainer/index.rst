hopwise.trainer
===============

.. py:module:: hopwise.trainer


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/trainer/hf_path_trainer/index
   /autoapi/hopwise/trainer/hyper_tuning/index
   /autoapi/hopwise/trainer/trainer/index


Classes
-------

.. autoapisummary::

   hopwise.trainer.HyperTuning
   hopwise.trainer.Trainer
   hopwise.trainer.KGTrainer
   hopwise.trainer.KGATTrainer
   hopwise.trainer.S3RecTrainer
   hopwise.trainer.TPRecTrainer
   hopwise.trainer.MKRTrainer
   hopwise.trainer.TraditionalTrainer
   hopwise.trainer.DecisionTreeTrainer
   hopwise.trainer.XGBoostTrainer
   hopwise.trainer.LightGBMTrainer
   hopwise.trainer.RaCTTrainer
   hopwise.trainer.RecVAETrainer
   hopwise.trainer.NCLTrainer
   hopwise.trainer.PEARLMfromscratchTrainer
   hopwise.trainer.HFPathLanguageModelingTrainer
   hopwise.trainer.KGGLMTrainer


Package Contents
----------------

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



.. py:class:: Trainer(config, model)

   Bases: :py:obj:`AbstractTrainer`


   The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
   functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   resume_checkpoint() and some other features helpful for model training and evaluation.

   Generally speaking, this class can serve most recommender system models, If the training process of the model is to
   simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
   pre-training and so on.

   Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
   for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
   `model` is the instantiated object of a Model Class.



   .. py:attribute:: logger


   .. py:attribute:: tensorboard


   .. py:attribute:: wandblogger


   .. py:attribute:: learner


   .. py:attribute:: learning_rate


   .. py:attribute:: epochs


   .. py:attribute:: eval_step


   .. py:attribute:: stopping_step


   .. py:attribute:: clip_grad_norm


   .. py:attribute:: valid_metric


   .. py:attribute:: valid_metric_bigger


   .. py:attribute:: test_batch_size


   .. py:attribute:: gpu_available


   .. py:attribute:: device


   .. py:attribute:: checkpoint_dir


   .. py:attribute:: enable_amp


   .. py:attribute:: enable_scaler


   .. py:attribute:: saved_model_file


   .. py:attribute:: weight_decay


   .. py:attribute:: start_epoch
      :value: 0



   .. py:attribute:: cur_step
      :value: 0



   .. py:attribute:: best_valid_score


   .. py:attribute:: best_valid_result
      :value: None



   .. py:attribute:: train_loss_dict


   .. py:attribute:: optimizer


   .. py:attribute:: eval_type


   .. py:attribute:: eval_collector


   .. py:attribute:: evaluator


   .. py:method:: _build_optimizer(**kwargs)

      Init the Optimizer

      :param params: The parameters to be optimized.
                     Defaults to ``self.model.parameters()``.
      :type params: torch.nn.Parameter, optional
      :param learner: The name of used optimizer. Defaults to ``self.learner``.
      :type learner: str, optional
      :param learning_rate: Learning rate. Defaults to ``self.learning_rate``.
      :type learning_rate: float, optional
      :param weight_decay: The L2 regularization weight. Defaults to ``self.weight_decay``.
      :type weight_decay: float, optional

      :returns: the optimizer
      :rtype: torch.optim



   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



   .. py:method:: _valid_epoch(valid_data, show_progress=False)

      Valid the model with valid data

      :param valid_data: the valid data.
      :type valid_data: DataLoader
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: valid score
                dict: valid result
      :rtype: float



   .. py:method:: _save_checkpoint(epoch, verbose=True, **kwargs)

      Store the model parameters information and training information.

      :param epoch: the current epoch id
      :type epoch: int



   .. py:method:: resume_checkpoint(resume_file)

      Load the model parameters information and training information.

      :param resume_file: the checkpoint file
      :type resume_file: file



   .. py:method:: _check_nan(loss)


   .. py:method:: _generate_train_loss_output(epoch_idx, s_time, e_time, losses)


   .. py:method:: _add_train_loss_to_tensorboard(epoch_idx, losses, tag='Loss/Train')


   .. py:method:: _add_hparam_to_tensorboard(best_valid_result)


   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



   .. py:method:: _batch_eval(batched_data, tot_item_num, neg_sampling=False, item_tensor=None)


   .. py:method:: _full_sort_batch_eval(batched_data, tot_item_num, item_tensor)


   .. py:method:: _neg_sample_batch_eval(batched_data, tot_item_num=None)


   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.

      :param eval_data: the eval data
      :type eval_data: DataLoader
      :param load_best_model: whether load the best model in the training process, default: True.
                              It should be set True, if users want to test the model after training.
      :type load_best_model: bool, optional
      :param model_file: the saved model file, default: None. If users want to test the previously
                         trained model file, they can set this parameter.
      :type model_file: str, optional
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: eval result, key is the eval metric and value in the corresponding metric value.
      :rtype: collections.OrderedDict



   .. py:method:: _map_reduce(result, num_sample)


   .. py:method:: _split_predict(interaction, batch_size)


.. py:class:: KGTrainer(config, model)

   Bases: :py:obj:`Trainer`


   KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
   recommendation related task and knowledge related task alternately.



   .. py:attribute:: train_rec_step


   .. py:attribute:: train_kg_step


   .. py:attribute:: best_valid_score_lp


   .. py:attribute:: best_valid_result_lp
      :value: None



   .. py:attribute:: cur_step_lp
      :value: 0



   .. py:attribute:: tail_tensor
      :value: None



   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



   .. py:method:: _valid_epoch(valid_data, show_progress=False)

      Valid the model with valid data

      :param valid_data: the valid data.
      :type valid_data: Dataloader, list[Dataloader]
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: valid score
                dict: valid result
      :rtype: float



   .. py:method:: _batch_eval(batched_data, tot_target_num, neg_sampling=False, task=None, target_tensor=None)


   .. py:method:: _full_sort_batch_eval(batched_data, full_sort_predict_fn, predict_fn, column_tensor, tot_column_num)


   .. py:method:: _split_predict_fn(interaction, batch_size, predict_fn)


   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.

      :param eval_data: the eval data.
      :type eval_data: Dataloader, list[Dataloader]
      :param load_best_model: whether load the best model in the training process, default: True.
                              It should be set True, if users want to test the model after training.
      :type load_best_model: bool, optional
      :param model_file: the saved model file, default: None. If users want to test the previously
                         trained model file, they can set this parameter.
      :type model_file: str, optional
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: eval result, key is the eval metric and value in the corresponding metric value.
      :rtype: collections.OrderedDict



   .. py:method:: evaluate_data_loop(eval_data, task, tot_target_num, target_tensor, show_progress=True)


   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



.. py:class:: KGATTrainer(config, model)

   Bases: :py:obj:`Trainer`


   KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method.


   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



.. py:class:: S3RecTrainer(config, model)

   Bases: :py:obj:`PretrainTrainer`


   S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
   It includes two training stages: pre-training ang fine-tuning.



   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



.. py:class:: TPRecTrainer(config, model)

   Bases: :py:obj:`PretrainTrainer`


   TPRecTrainer is designed for TPRec, which is a knowledge-aware recommendation method.


   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.

      :param eval_data: the eval data
      :type eval_data: DataLoader
      :param load_best_model: whether load the best model in the training process, default: True.
                              It should be set True, if users want to test the model after training.
      :type load_best_model: bool, optional
      :param model_file: the saved model file, default: None. If users want to test the previously
                         trained model file, they can set this parameter.
      :type model_file: str, optional
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: eval result, key is the eval metric and value in the corresponding metric value.
      :rtype: collections.OrderedDict



   .. py:method:: _full_sort_batch_eval(batched_data, tot_item_num, item_tensor)


.. py:class:: MKRTrainer(config, model)

   Bases: :py:obj:`Trainer`


   MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method.


   .. py:attribute:: kge_interval


   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



.. py:class:: TraditionalTrainer(config, model)

   Bases: :py:obj:`Trainer`


   TraditionalTrainer is designed for Traditional model(Pop,ItemKNN),
   which set the epoch to 1 whatever the config.


   .. py:attribute:: epochs
      :value: 1



.. py:class:: DecisionTreeTrainer(config, model)

   Bases: :py:obj:`AbstractTrainer`


   DecisionTreeTrainer is designed for DecisionTree model.


   .. py:attribute:: logger


   .. py:attribute:: tensorboard


   .. py:attribute:: label_field


   .. py:attribute:: convert_token_to_onehot


   .. py:attribute:: eval_type


   .. py:attribute:: epochs


   .. py:attribute:: eval_step


   .. py:attribute:: valid_metric


   .. py:attribute:: eval_collector


   .. py:attribute:: evaluator


   .. py:attribute:: checkpoint_dir


   .. py:attribute:: temp_file


   .. py:attribute:: temp_best_file


   .. py:attribute:: saved_model_file


   .. py:attribute:: stopping_step


   .. py:attribute:: valid_metric_bigger


   .. py:attribute:: cur_step
      :value: 0



   .. py:attribute:: best_valid_score


   .. py:attribute:: best_valid_result
      :value: None



   .. py:method:: _interaction_to_sparse(dataloader)

      Convert data format from interaction to sparse or numpy

      :param dataloader: DecisionTreeDataLoader dataloader.
      :type dataloader: DecisionTreeDataLoader

      :returns: data.
                interaction_np[self.label_field] (numpy): label.
      :rtype: cur_data (sparse or numpy)



   .. py:method:: _interaction_to_lib_datatype(dataloader)


   .. py:method:: _valid_epoch(valid_data)

      Args:
      valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.



   .. py:method:: _save_checkpoint(epoch)

      Store the model parameters information and training information.

      :param epoch: the current epoch id
      :type epoch: int



   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data.



   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)
      :abstractmethod:


      Evaluate the model based on the eval data.



   .. py:method:: _train_at_once(train_data, valid_data)
      :abstractmethod:



.. py:class:: XGBoostTrainer(config, model)

   Bases: :py:obj:`DecisionTreeTrainer`


   XGBoostTrainer is designed for XGBOOST.


   .. py:attribute:: xgb


   .. py:attribute:: boost_model


   .. py:attribute:: silent


   .. py:attribute:: nthread


   .. py:attribute:: params


   .. py:attribute:: num_boost_round


   .. py:attribute:: evals
      :value: ()



   .. py:attribute:: early_stopping_rounds


   .. py:attribute:: evals_result


   .. py:attribute:: verbose_eval


   .. py:attribute:: callbacks
      :value: None



   .. py:attribute:: deval
      :value: None



   .. py:method:: _interaction_to_lib_datatype(dataloader)

      Convert data format from interaction to DMatrix

      :param dataloader: xgboost dataloader.
      :type dataloader: DecisionTreeDataLoader

      :returns: Data in the form of 'DMatrix'.
      :rtype: DMatrix



   .. py:method:: _train_at_once(train_data, valid_data)

      Args:
      train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
      valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.



   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.



.. py:class:: LightGBMTrainer(config, model)

   Bases: :py:obj:`DecisionTreeTrainer`


   LightGBMTrainer is designed for LightGBM.


   .. py:attribute:: lgb


   .. py:attribute:: params


   .. py:attribute:: num_boost_round


   .. py:attribute:: evals
      :value: ()



   .. py:method:: _interaction_to_lib_datatype(dataloader)

      Convert data format from interaction to Dataset

      :param dataloader: xgboost dataloader.
      :type dataloader: DecisionTreeDataLoader

      :returns: Data in the form of 'lgb.Dataset'.
      :rtype: dataset(lgb.Dataset)



   .. py:method:: _train_at_once(train_data, valid_data)

      Args:
      train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
      valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.



   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.



.. py:class:: RaCTTrainer(config, model)

   Bases: :py:obj:`PretrainTrainer`


   RaCTTrainer is designed for RaCT, which is an actor-critic reinforcement learning based general recommenders.
   It includes three training stages: actor pre-training, critic pre-training and actor-critic training.



   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



.. py:class:: RecVAETrainer(config, model)

   Bases: :py:obj:`Trainer`


   RecVAETrainer is designed for RecVAE, which is a general recommender.


   .. py:attribute:: n_enc_epochs


   .. py:attribute:: n_dec_epochs


   .. py:attribute:: optimizer_encoder


   .. py:attribute:: optimizer_decoder


   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch.

      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns:

                The sum of loss returned by all batches in this epoch.
                    If the loss in each batch contains multiple parts and the model
                    returns these multiple parts loss instead of the sum of loss, it
                    will return a tuple which includes the sum of loss in each part.
      :rtype: float or tuple



.. py:class:: NCLTrainer(config, model)

   Bases: :py:obj:`Trainer`


   The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
   functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   resume_checkpoint() and some other features helpful for model training and evaluation.

   Generally speaking, this class can serve most recommender system models, If the training process of the model is to
   simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
   pre-training and so on.

   Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
   for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
   `model` is the instantiated object of a Model Class.



   .. py:attribute:: num_m_step


   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data.
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



   .. py:method:: _train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)

      Train the model in an epoch
      :param train_data: The train data.
      :type train_data: DataLoader
      :param epoch_idx: The current epoch id.
      :type epoch_idx: int
      :param loss_func: The loss function of :attr:`model`. If it is ``None``, the loss function will be
                        :attr:`self.model.calculate_loss`. Defaults to ``None``.
      :type loss_func: function
      :param show_progress: Show the progress of training epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
                multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
                tuple which includes the sum of loss in each part.
      :rtype: float/tuple



.. py:class:: PEARLMfromscratchTrainer(config, model)

   Bases: :py:obj:`ExplainableTrainer`


   ExplainableTrainer is designed for explainable recommendation methods.


   .. py:attribute:: path_generation_args


   .. py:method:: _full_sort_batch_eval(batched_data, tot_item_num, item_tensor)


.. py:class:: HFPathLanguageModelingTrainer(config, model)

   Bases: :py:obj:`ExplainableTrainer`


   HFPathLanguageModelingTrainer is designed for path-based knowledge-aware recommendation methods.
   It is specifically designed to communicate with the Hugging Face Trainer to use language models and functionalities
   as tokenizers and beam search.


   .. py:attribute:: HOPWISE_SAVE_PATH_SUFFIX
      :value: 'hopwise-'



   .. py:attribute:: HUGGINGFACE_SAVE_PATH_SUFFIX
      :value: 'huggingface-'



   .. py:attribute:: path_generation_args


   .. py:attribute:: saved_model_file


   .. py:method:: prepare_hf_args(**kwargs)


   .. py:method:: init_hf_trainer(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, hf_callbacks=None, callback_fn=None, training_args=None)


   .. py:property:: processing_class


   .. py:method:: _save_checkpoint(epoch, verbose=True, **kwargs)

      Store the model parameters information and training information.

      :param epoch: the current epoch id
      :type epoch: int



   .. py:method:: resume_checkpoint(resume_file)

      Load the model parameters and training information based on the directory name,
      and navigate into subdirectories if necessary.
      Also handles both HuggingFace and Hopwise formats by reading corresponding files.

      :param resume_file: the path to the directory containing the checkpoint files or subdirectories
      :type resume_file: str



   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



   .. py:method:: _full_sort_batch_eval(batched_data, tot_item_num, item_tensor)


   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.

      :param eval_data: the eval data
      :type eval_data: DataLoader
      :param load_best_model: whether load the best model in the training process, default: True.
                              It should be set True, if users want to test the model after training.
      :type load_best_model: bool, optional
      :param model_file: the saved model file, default: None. If users want to test the previously
                         trained model file, they can set this parameter.
      :type model_file: str, optional
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: eval result, key is the eval metric and value in the corresponding metric value.
      :rtype: collections.OrderedDict



.. py:class:: KGGLMTrainer(config, model)

   Bases: :py:obj:`HFPathLanguageModelingTrainer`, :py:obj:`PretrainTrainer`


   KGGLMTrainer is designed for KGGLM, which is a path-based language model for knowledge-aware recommendation.
   It includes two training stages: link prediction pre-training and recommendation path generation fine-tuning.


   .. py:method:: _get_pretrained_model_path(epoch_label=None)


   .. py:method:: pretrain(train_data, verbose=True, show_progress=False)


   .. py:method:: evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)

      Evaluate the model based on the eval data.

      :param eval_data: the eval data
      :type eval_data: DataLoader
      :param load_best_model: whether load the best model in the training process, default: True.
                              It should be set True, if users want to test the model after training.
      :type load_best_model: bool, optional
      :param model_file: the saved model file, default: None. If users want to test the previously
                         trained model file, they can set this parameter.
      :type model_file: str, optional
      :param show_progress: Show the progress of evaluate epoch. Defaults to ``False``.
      :type show_progress: bool

      :returns: eval result, key is the eval metric and value in the corresponding metric value.
      :rtype: collections.OrderedDict



   .. py:method:: fit(train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)

      Train the model based on the train data and the valid data.

      :param train_data: the train data
      :type train_data: DataLoader
      :param valid_data: the valid data, default: None.
                         If it's None, the early_stopping is invalid.
      :type valid_data: DataLoader, optional
      :param verbose: whether to write training and evaluation information to logger, default: True
      :type verbose: bool, optional
      :param saved: whether to save the model parameters, default: True
      :type saved: bool, optional
      :param show_progress: Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
      :type show_progress: bool
      :param callback_fn: Optional callback function executed at end of epoch.
                          Includes (epoch_idx, valid_score) input arguments.
      :type callback_fn: callable

      :returns: best valid score and best valid result. If valid_data is None, it returns (-1, None)
      :rtype: (float, dict)



