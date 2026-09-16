hopwise.model.path_language_modeling_recommender.kgglm
======================================================

.. py:module:: hopwise.model.path_language_modeling_recommender.kgglm

.. autoapi-nested-parse::

   KGGLM
   ##################################################
   Reference:
       Balloccu et al. "KGGLM: A Generative Language Model for Generalizable Knowledge
       Graph Representation Learning in Recommendation." in RecSys 2024.

   Reference code:
       https://github.com/mirkomarras/kgglm



Classes
-------

.. autoapisummary::

   hopwise.model.path_language_modeling_recommender.kgglm.KGGLM


Module Contents
---------------

.. py:class:: KGGLM(config, dataset)

   Bases: :py:obj:`hopwise.model.path_language_modeling_recommender.pearlm.PEARLM`


   PEARLM is a path-language-modeling recommender. It learns the sequence of entity-relation triplets
   as paths extracted from a knowledge graph. It is trained to predict the next token in a sequence of tokens
   representing a path. The model extends PLM by adding a constrained graph decoding mechanism to ensure that
   the generated paths are valid according to the knowledge graph structure. The model can be used for
   explainable recommendation by generating paths that explain the recommendations made by the model.


   .. py:attribute:: TRAIN_STAGES
      :value: ['pretrain', 'finetune']



   .. py:attribute:: train_stage


   .. py:attribute:: pre_model_path


