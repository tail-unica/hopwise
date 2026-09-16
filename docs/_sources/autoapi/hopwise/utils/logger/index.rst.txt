hopwise.utils.logger
====================

.. py:module:: hopwise.utils.logger

.. autoapi-nested-parse::

   hopwise.utils.logger
   ###############################



Attributes
----------

.. autoapisummary::

   hopwise.utils.logger._progress_bar
   hopwise.utils.logger.log_colors_config


Classes
-------

.. autoapisummary::

   hopwise.utils.logger.ProgressBar
   hopwise.utils.logger.RemoveColorFilter


Functions
---------

.. autoapisummary::

   hopwise.utils.logger.progress_bar
   hopwise.utils.logger.set_color
   hopwise.utils.logger.init_logger


Module Contents
---------------

.. py:data:: _progress_bar
   :value: None


.. py:class:: ProgressBar(progress_bar_rich=True)

   .. py:attribute:: progress_bar


   .. py:method:: __call__(*args, **kwargs)


.. py:function:: progress_bar(*args, **kwargs)

.. py:data:: log_colors_config

.. py:class:: RemoveColorFilter(name='')

   Bases: :py:obj:`logging.Filter`


   Filter instances are used to perform arbitrary filtering of LogRecords.

   Loggers and Handlers can optionally use Filter instances to filter
   records as desired. The base filter class only allows events which are
   below a certain point in the logger hierarchy. For example, a filter
   initialized with "A.B" will allow events logged by loggers "A.B",
   "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
   initialized with the empty string, all events are passed.


   .. py:method:: filter(record)

      Determine if the specified record is to be logged.

      Returns True if the record should be logged, or False otherwise.
      If deemed appropriate, the record may be modified in-place.



.. py:function:: set_color(log, color, highlight=True, progress=False)

.. py:function:: init_logger(config)

   A logger that can show a message on standard output and write it into the
   file named `filename` simultaneously.
   All the message that you want to log MUST be str.

   :param config: An instance object of Config, used to record parameter information.
   :type config: Config

   .. rubric:: Example

   >>> logger = logging.getLogger(config)
   >>> logger.debug(train_state)
   >>> logger.info(train_result)


