hopwise.utils.wandblogger
=========================

.. py:module:: hopwise.utils.wandblogger

.. autoapi-nested-parse::

   hopwise.utils.wandblogger
   ################################



Classes
-------

.. autoapisummary::

   hopwise.utils.wandblogger.WandbLogger


Module Contents
---------------

.. py:class:: WandbLogger(config)

   WandbLogger to log metrics to Weights and Biases.


   .. py:attribute:: config


   .. py:attribute:: log_wandb


   .. py:method:: setup()


   .. py:method:: log_metrics(metrics, head='train', commit=True)


   .. py:method:: log_eval_metrics(metrics, head='eval')


   .. py:method:: _set_steps()


   .. py:method:: _add_head_to_metrics(metrics, head)


