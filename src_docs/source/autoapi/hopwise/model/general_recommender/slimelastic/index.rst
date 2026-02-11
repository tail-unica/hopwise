hopwise.model.general_recommender.slimelastic
=============================================

.. py:module:: hopwise.model.general_recommender.slimelastic

.. autoapi-nested-parse::

   SLIMElastic
   ################################################
   Reference:
       Xia Ning et al. "SLIM: Sparse Linear Methods for Top-N Recommender Systems." in ICDM 2011.
   Reference code:
       https://github.com/KarypisLab/SLIM
       https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/master/SLIM_ElasticNet/SLIMElasticNetRecommender.py



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.slimelastic.SLIMElastic


Module Contents
---------------

.. py:class:: SLIMElastic(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   SLIMElastic is a sparse linear method for top-K recommendation, which learns
   a sparse aggregation coefficient matrix by solving an L1-norm and L2-norm
   regularized optimization problem.



   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: hide_item


   .. py:attribute:: alpha


   .. py:attribute:: l1_ratio


   .. py:attribute:: positive_only


   .. py:attribute:: dummy_param


   .. py:attribute:: interaction_matrix


   .. py:attribute:: item_similarity


   .. py:attribute:: other_parameter_name
      :value: ['interaction_matrix', 'item_similarity']



   .. py:method:: forward()


   .. py:method:: calculate_loss(interaction)

      Calculate the training loss for a batch data.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Training loss, shape: []
      :rtype: torch.Tensor



   .. py:method:: predict(interaction)

      Predict the scores between users and items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and items, shape: [batch_size]
      :rtype: torch.Tensor



   .. py:method:: full_sort_predict(interaction)

      Full sort prediction function.
      Given users, calculate the scores between users and all candidate items.

      :param interaction: Interaction class of the batch.
      :type interaction: Interaction

      :returns: Predicted scores for given users and all candidate items,
                shape: [n_batch_users * n_candidate_items]
      :rtype: torch.Tensor



