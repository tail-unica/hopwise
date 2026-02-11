hopwise.model.general_recommender.nceplrec
==========================================

.. py:module:: hopwise.model.general_recommender.nceplrec

.. autoapi-nested-parse::

   NCE-PLRec
   ######################################
   Reference:
       Ga Wu, et al. "Noise Contrastive Estimation for One-Class Collaborative Filtering" in SIGIR 2019.
   Reference code:
       https://github.com/wuga214/NCE_Projected_LRec



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.nceplrec.NCEPLRec


Module Contents
---------------

.. py:class:: NCEPLRec(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   This is a abstract general recommender. All the general model should implement this class.
   The base general recommender class provide the basic dataset and parameters information.


   .. py:attribute:: input_type


   .. py:attribute:: dummy_param


   .. py:attribute:: user_embeddings


   .. py:attribute:: item_embeddings


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



