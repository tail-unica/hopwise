hopwise.model.knowledge_graph_embedding_recommender.convkb
==========================================================

.. py:module:: hopwise.model.knowledge_graph_embedding_recommender.convkb

.. autoapi-nested-parse::

   ConvKB
   ##################################################
   Reference:
       Nguyen et al. "A Novel Embedding Model for Knowledge Base Completion Based on
       Convolutional Neural Network." in NAACL 2018.

   Reference code:
       https://github.com/torchkge-team/torchkge



Classes
-------

.. autoapisummary::

   hopwise.model.knowledge_graph_embedding_recommender.convkb.ConvKB


Module Contents
---------------

.. py:class:: ConvKB(config, dataset)

   Bases: :py:obj:`hopwise.model.abstract_recommender.KnowledgeRecommender`


   ConvKB: The main differences from ConvE are that when scoring h, r and t,
   it concatenates them into a d x 3 matrix. This output undergoes convolution
   by a set of omega of T filters of shape 1x3, resulting in a Tx3 feature map.
   This feature map goes through a dense layer with one neuron and weights W, resulting in the final score.

   .. note:: In this version, we sample recommender data and knowledge data separately, and put them together for training.


   .. py:attribute:: input_type


   .. py:attribute:: embedding_size


   .. py:attribute:: device


   .. py:attribute:: out_channels


   .. py:attribute:: kernel_size


   .. py:attribute:: drop_prob


   .. py:attribute:: lmbda


   .. py:attribute:: ui_relation


   .. py:attribute:: user_embedding


   .. py:attribute:: entity_embedding


   .. py:attribute:: relation_embedding


   .. py:attribute:: conv1_bn


   .. py:attribute:: conv_layer


   .. py:attribute:: conv2_bn


   .. py:attribute:: dropout


   .. py:attribute:: non_linearity


   .. py:attribute:: fc_layer


   .. py:attribute:: loss


   .. py:attribute:: reg


   .. py:method:: forward(head, relation, tail)


   .. py:method:: _get_regularization(head, relation, tail)


   .. py:method:: _get_rec_embeddings(user, positive_items, negative_items)


   .. py:method:: _get_kg_embeddings(head, relation, positive_tails, negative_tails)


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



   .. py:method:: predict_kg(interaction)


