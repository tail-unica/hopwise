hopwise.model.general_recommender.itemknn
=========================================

.. py:module:: hopwise.model.general_recommender.itemknn

.. autoapi-nested-parse::

   ItemKNN
   ################################################
   Reference:
       Aiolli,F et al. Efficient top-n recommendation for very large scale binary rated datasets.
       In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.



Classes
-------

.. autoapisummary::

   hopwise.model.general_recommender.itemknn.ComputeSimilarity
   hopwise.model.general_recommender.itemknn.ItemKNN


Module Contents
---------------

.. py:class:: ComputeSimilarity(dataMatrix, topk=100, shrink=0, normalize=True)

   .. py:attribute:: shrink
      :value: 0



   .. py:attribute:: normalize
      :value: True



   .. py:attribute:: TopK


   .. py:attribute:: dataMatrix


   .. py:method:: compute_similarity(method, block_size=100)

      Compute the similarity for the given dataset

      :param method: Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
      :type method: str
      :param block_size: divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                         otherwise, divide matrix to :math:`n\_columns \div block\_size`.
      :type block_size: int

      :returns: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
                else, the shape is [number of items, neigh_num].
                scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
                else, the shape is [self.n_columns, self.n_columns].
      :rtype: list



.. py:class:: ItemKNN(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.GeneralRecommender`


   ItemKNN is a basic model that compute item similarity with the interaction matrix.


   .. py:attribute:: input_type


   .. py:attribute:: type


   .. py:attribute:: k


   .. py:attribute:: shrink


   .. py:attribute:: interaction_matrix


   .. py:attribute:: pred_mat


   .. py:attribute:: fake_loss


   .. py:attribute:: other_parameter_name
      :value: ['w', 'pred_mat']



   .. py:method:: forward(user, item)


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



