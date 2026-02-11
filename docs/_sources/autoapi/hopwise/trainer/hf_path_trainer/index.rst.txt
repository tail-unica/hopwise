hopwise.trainer.hf_path_trainer
===============================

.. py:module:: hopwise.trainer.hf_path_trainer


Classes
-------

.. autoapisummary::

   hopwise.trainer.hf_path_trainer.HFPathTrainer
   hopwise.trainer.hf_path_trainer.HopwiseCallback


Module Contents
---------------

.. py:class:: HFPathTrainer(model, callbacks, train_data=None, args=None, tokenizer=None)

   Bases: :py:obj:`transformers.Trainer`


   A HuggingFace Trainer that integrates with Hopwise for training and evaluation.


   .. py:method:: evaluate(**kwargs)


.. py:class:: HopwiseCallback(hopwise_trainer, train_data=None, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None, model=None, model_name=None)

   Bases: :py:obj:`transformers.TrainerCallback`


   It handles the training and evaluation communication with the hopwise and HuggingFace trainers.


   .. py:attribute:: model
      :value: None



   .. py:attribute:: model_name
      :value: None



   .. py:attribute:: hopwise_trainer


   .. py:attribute:: train_data
      :value: None



   .. py:attribute:: valid_data
      :value: None



   .. py:attribute:: verbose
      :value: True



   .. py:attribute:: saved
      :value: True



   .. py:attribute:: show_progress
      :value: False



   .. py:attribute:: callback_fn
      :value: None



   .. py:method:: on_train_begin(args, state, control, **kwargs)


   .. py:method:: on_train_end(args, state, control, **kwargs)


   .. py:method:: on_epoch_begin(args, state, control, **kwargs)


   .. py:method:: on_epoch_end(args, state, control, **kwargs)


   .. py:method:: on_step_end(args, state, control, **kwargs)


   .. py:method:: on_evaluate(args, state, control, **kwargs)


