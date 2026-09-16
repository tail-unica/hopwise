hopwise.model.sequential_recommender.fpmc
=========================================

.. py:module:: hopwise.model.sequential_recommender.fpmc

.. autoapi-nested-parse::

   FPMC
   ################################################

   Reference:
       Steffen Rendle et al. "Factorizing Personalized Markov Chains for Next-Basket Recommendation." in WWW 2010.



Classes
-------

.. autoapisummary::

   hopwise.model.sequential_recommender.fpmc.FPMC


Module Contents
---------------

.. py:class:: FPMC(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.SequentialRecommender`


   The FPMC model is mainly used in the recommendation system to predict the possibility of
   unknown items arousing user interest, and to discharge the item recommendation list.

   .. note::

      In order that the generation method we used is common to other sequential models,
      We set the size of the basket mentioned in the paper equal to 1.
      For comparison with other models, the loss function used is BPR.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: loss_type


   .. py:attribute:: n_users


   .. py:attribute:: UI_emb


   .. py:attribute:: IU_emb


   .. py:attribute:: LI_emb


   .. py:attribute:: IL_emb


   .. py:method:: _init_weights(module)


   .. py:method:: forward(user, item_seq, item_seq_len, next_item)


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



