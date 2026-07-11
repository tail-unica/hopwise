hopwise.utils.enum_type
=======================

.. py:module:: hopwise.utils.enum_type

.. autoapi-nested-parse::

   hopwise.utils.enum_type
   #######################



Classes
-------

.. autoapisummary::

   hopwise.utils.enum_type.ModelType
   hopwise.utils.enum_type.KGDataLoaderState
   hopwise.utils.enum_type.KnowledgeEvaluationType
   hopwise.utils.enum_type.EvaluatorType
   hopwise.utils.enum_type.InputType
   hopwise.utils.enum_type.FeatureType
   hopwise.utils.enum_type.FeatureSource
   hopwise.utils.enum_type.PathLanguageModelingTokenType


Module Contents
---------------

.. py:class:: ModelType

   Bases: :py:obj:`enum.Enum`


   Type of models.

   - ``GENERAL``: General Recommendation
   - ``SEQUENTIAL``: Sequential Recommendation
   - ``CONTEXT``: Context-aware Recommendation
   - ``KNOWLEDGE``: Knowledge-based Recommendation
   - ``PATH_LANGUAGE_MODELING``: Path Language Modeling Recommendation


   .. py:attribute:: GENERAL
      :value: 1



   .. py:attribute:: SEQUENTIAL
      :value: 2



   .. py:attribute:: CONTEXT
      :value: 3



   .. py:attribute:: KNOWLEDGE
      :value: 4



   .. py:attribute:: TRADITIONAL
      :value: 5



   .. py:attribute:: DECISIONTREE
      :value: 6



   .. py:attribute:: PATH_LANGUAGE_MODELING
      :value: 7



.. py:class:: KGDataLoaderState

   Bases: :py:obj:`enum.Enum`


   States for Knowledge-based DataLoader.

   - ``RSKG``: Return both knowledge graph information and user-item interaction information.
   - ``RS``: Only return the user-item interaction.
   - ``KG``: Only return the triplets with negative examples in a knowledge graph.


   .. py:attribute:: RSKG
      :value: 1



   .. py:attribute:: RS
      :value: 2



   .. py:attribute:: KG
      :value: 3



.. py:class:: KnowledgeEvaluationType

   Bases: :py:obj:`enum.Enum`


   Type of evaluation task: Recommendation or Link Prediction

   - ``REC``: Evaluate on Recommendation
   - ``LP``:  Evaluate on Link Prediction


   .. py:attribute:: REC
      :value: 1



   .. py:attribute:: LP
      :value: 2



   .. py:method:: __str__()


.. py:class:: EvaluatorType

   Bases: :py:obj:`enum.Enum`


   Type for evaluation metrics.

   - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
   - ``VALUE``: Value-based metrics like AUC, etc.


   .. py:attribute:: RANKING
      :value: 1



   .. py:attribute:: VALUE
      :value: 2



.. py:class:: InputType

   Bases: :py:obj:`enum.Enum`


   Type of Models' input.

   - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
   - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
   - ``LISTWISE``: List-wise input, like ``uid, [iid1, iid2, ...]``.
   - ``PATHWISE``: KG Path-wise input, like ``uid, pos_iid, eid1, eid2, next_pos_iid``.
   - ``USERWISE``: User-wise input, like ``uid0, uid1, ...., uidn``.


   .. py:attribute:: POINTWISE
      :value: 1



   .. py:attribute:: PAIRWISE
      :value: 2



   .. py:attribute:: LISTWISE
      :value: 3



   .. py:attribute:: PATHWISE
      :value: 4



   .. py:attribute:: USERWISE
      :value: 5



.. py:class:: FeatureType

   Bases: :py:obj:`enum.Enum`


   Type of features.

   - ``TOKEN``: Token features like user_id and item_id.
   - ``FLOAT``: Float features like rating and timestamp.
   - ``TOKEN_SEQ``: Token sequence features like review.
   - ``FLOAT_SEQ``: Float sequence features like pretrained vector.


   .. py:attribute:: TOKEN
      :value: 'token'



   .. py:attribute:: FLOAT
      :value: 'float'



   .. py:attribute:: TOKEN_SEQ
      :value: 'token_seq'



   .. py:attribute:: FLOAT_SEQ
      :value: 'float_seq'



.. py:class:: FeatureSource

   Bases: :py:obj:`enum.Enum`


   Source of features.

   - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
   - ``USER``: Features from ``.user`` (other than ``user_id``).
   - ``ITEM``: Features from ``.item`` (other than ``item_id``).
   - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
   - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
   - ``KG``: Features from ``.kg``.
   - ``NET``: Features from ``.net``.


   .. py:attribute:: INTERACTION
      :value: 'inter'



   .. py:attribute:: USER
      :value: 'user'



   .. py:attribute:: ITEM
      :value: 'item'



   .. py:attribute:: USER_ID
      :value: 'user_id'



   .. py:attribute:: ITEM_ID
      :value: 'item_id'



   .. py:attribute:: KG
      :value: 'kg'



   .. py:attribute:: NET
      :value: 'net'



.. py:class:: PathLanguageModelingTokenType(token, token_id)

   Bases: :py:obj:`enum.Enum`


   Type of tokens in paths for Path Language Modeling.

   - ``SPECIAL``: Special tokens, like start and end of a path.
   - ``ENTITY``: Entity tokens.
   - ``RELATION``: Relation tokens.
   - ``USER``: User tokens.
   - ``ITEM``: Item tokens.


   .. py:attribute:: SPECIAL
      :value: ('S', 0)



   .. py:attribute:: ENTITY
      :value: ('E', 1)



   .. py:attribute:: RELATION
      :value: ('R', 2)



   .. py:attribute:: USER
      :value: ('U', 3)



   .. py:attribute:: ITEM
      :value: ('I', 4)



   .. py:attribute:: token


   .. py:attribute:: token_id


   .. py:method:: __str__()


