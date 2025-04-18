# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""hopwise.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``GENERAL``: General Recommendation
    - ``SEQUENTIAL``: Sequential Recommendation
    - ``CONTEXT``: Context-aware Recommendation
    - ``KNOWLEDGE``: Knowledge-based Recommendation
    - ``PATH_LANGUAGE_MODELING``: Path Language Modeling Recommendation
    """

    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6
    PATH_LANGUAGE_MODELING = 7


class KGDataLoaderState(Enum):
    """States for Knowledge-based DataLoader.

    - ``RSKG``: Return both knowledge graph information and user-item interaction information.
    - ``RS``: Only return the user-item interaction.
    - ``KG``: Only return the triplets with negative examples in a knowledge graph.
    """

    RSKG = 1
    RS = 2
    KG = 3


class KnowledgeEvaluationType(Enum):
    """Type of evaluation task: Recommendation or Link Prediction

    - ``REC``: Evaluate on Recommendation
    - ``LP``:  Evaluate on Link Prediction
    """

    REC = 1
    LP = 2

    def __str__(self):
        _descriptions = {KnowledgeEvaluationType.REC: "recommendation", KnowledgeEvaluationType.LP: "link prediction"}

        return _descriptions[self]


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """

    RANKING = 1
    VALUE = 2


class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    - ``LISTWISE``: List-wise input, like ``uid, [iid1, iid2, ...]``.
    - ``PATHWISE``: KG Path-wise input, like ``uid, pos_iid, eid1, eid2, next_pos_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3
    PATHWISE = 4


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = "token"
    FLOAT = "float"
    TOKEN_SEQ = "token_seq"
    FLOAT_SEQ = "float_seq"


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    - ``KG``: Features from ``.kg``.
    - ``NET``: Features from ``.net``.
    """

    INTERACTION = "inter"
    USER = "user"
    ITEM = "item"
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    KG = "kg"
    NET = "net"


class PathLanguageModelingTokenType(Enum):
    """Type of tokens in paths for Path Language Modeling.

    - ``USER``: User tokens.
    - ``ITEM``: Item tokens.
    - ``ENTITY``: Entity tokens.
    - ``RELATION``: Relation tokens.

    """

    USER = "U"
    ITEM = "I"
    ENTITY = "E"
    RELATION = "R"
