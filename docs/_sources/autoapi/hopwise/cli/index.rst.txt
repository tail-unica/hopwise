hopwise.cli
===========

.. py:module:: hopwise.cli


Attributes
----------

.. autoapisummary::

   hopwise.cli.console
   hopwise.cli.debug_message


Classes
-------

.. autoapisummary::

   hopwise.cli.HopwiseClickCommand


Functions
---------

.. autoapisummary::

   hopwise.cli.cli
   hopwise.cli.train
   hopwise.cli.evaluate
   hopwise.cli.benchmark
   hopwise.cli.tune
   hopwise.cli.models


Module Contents
---------------

.. py:class:: HopwiseClickCommand

   Bases: :py:obj:`click.Command`


   .. py:method:: parse_args(ctx, args)

      Override to filter out HopWise parameters before Click's validation



.. py:data:: console

.. py:data:: debug_message
   :value: Multiline-String

   .. raw:: html

      <details><summary>Show Value</summary>

   .. code-block:: python

      """[dim]
          Use --debug for full traceback or --rich-traceback for enhanced formatting. Please, be careful
          that it should be placed after hopwise command and before any subcommand, e.g.:
          hopwise train --debug [model] [dataset] [--config-files config1 config2 ...]
          hopwise train --rich-traceback [model] [dataset] [--config-files config1 config2 ...]
          [/dim]"""

   .. raw:: html

      </details>



.. py:function:: cli(ctx, debug, rich_traceback)

   🔮 HopWise - Advanced Knowledge Graph-Enhanced Recommendation System

   HopWise extends RecBole with knowledge graphs, path-based reasoning,
   and language modeling for explainable recommendations.


.. py:function:: train(ctx, model, dataset, config_files, nproc, checkpoint, ip, port, world_size, group_offset, proc_title)

   Train or evaluate a single model.


.. py:function:: evaluate(ctx, model, dataset, config_files, nproc, checkpoint, ip, port, world_size, group_offset, proc_title)

   Train or evaluate a single model.


.. py:function:: benchmark(models, dataset, config_files, valid_latex, test_latex, nproc, ip, port, world_size, group_offset, proc_title)

   Run scientific benchmark experiments across multiple models.

   Trains multiple models on the same dataset and generates comparative results
   in LaTeX table format for scientific publications. Ideal for reproducing
   paper results or conducting systematic model comparisons.

   .. rubric:: Example

   hopwise benchmark --models "BPR,LightGCN,KGAT" --dataset ml-100k --show-progress


.. py:function:: tune(ctx, params_file, config_files, output_path, display_file, max_evals, tool, study_name, algo, resume, proc_title)

   Run hyperparameter tuning.


.. py:function:: models(verbose, model_types)

   List available models.


