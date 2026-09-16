# @Time    : 2026/01/26
# @Author  : Emanuele Caddeo

# UPDATE
# @Time    :    2026/09/15
# @Author  :    Giacomo Medda
# @email   :    jackm.medda@gmail.com

import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pytest

from hopwise.config import Config
from hopwise.evaluator.register import metrics_dict

TOPK = [5, 10, 20]
DECIMAL_PLACE = 4
TOL = 10**-DECIMAL_PLACE


@pytest.fixture(scope="module")
def config():
    return Config("BPR", "ml-1m", config_dict={"topk": TOPK, "metric_decimal_place": DECIMAL_PLACE})


class _DataObject:
    """Minimal stand-in for the evaluator data struct, exposing only the keys a metric reads."""

    def __init__(self, store):
        self._store = store

    def get(self, key):
        return self._store[key]


def _compute(config, name, store):
    return metrics_dict[name](config).calculate_metric(_DataObject(store))


def _assert_metric(result, prefix, expected):
    """Check ``prefix@k`` for every k in ``TOPK``.

    ``expected`` is either a scalar, for metrics reporting the same value for every k, or a ``{k: value}`` dict.
    """
    if not isinstance(expected, dict):
        expected = dict.fromkeys(TOPK, expected)
    got = {k: float(result[f"{prefix}@{k}"]) for k in TOPK}
    assert got == pytest.approx(expected, abs=TOL)


def _mk_path(user, rec_item, *, link=("item", 10), shared=("entity", 999), rel_ids=(0, 1, 2, 3)):
    """Build a ``(user, rec_item, score, path)`` entry of ``rec.paths``.

    Nodes are ``(relation_id, node_type, node_id)`` triples: user -> link -> shared -> rec_item.
    Fields read by each metric:
      - LID: link id ``path[1][-1]``; LITD: link type ``path[1][1]``; LIR: both
      - SED: shared id ``path[-2][-1]``; SETD: shared type ``path[-2][1]``; SEP: both
      - PTD/PTC: path type ``path[-1][0]``, or ``path[-2][0]`` when the former is ``"self_loop"``
      - PPT: relation ids of ``path[1:]``
      - Fidelity: ``rec_item``
    """
    path = [
        (rel_ids[0], "user", user),
        (rel_ids[1], link[0], link[1]),
        (rel_ids[2], shared[0], shared[1]),
        (rel_ids[3], "item", rec_item),
    ]
    return (user, rec_item, 0.0, path)


def _typed_path(user, rec_item, path_type, *, self_loop=False):
    """Path whose type is ``path_type``, stored in the penultimate node when ``self_loop`` is set."""
    rel_ids = (0, 1, path_type, "self_loop") if self_loop else (0, 1, 2, path_type)
    return _mk_path(user, rec_item, rel_ids=rel_ids)


# -------------------------------------------------------------------------
# Fidelity
# -------------------------------------------------------------------------

FIDELITY_CASES = [
    pytest.param(
        [_mk_path(0, item) for item in (20, 21, 22, 23)],
        {5: 4 / 5, 10: 4 / 10, 20: 4 / 20},
        id="one_user_fewer_items_than_k",
    ),
    pytest.param(
        [_mk_path(0, item) for item in (20, 21, 22, 23, 24, 25)],
        {5: 1.0, 10: 6 / 10, 20: 6 / 20},
        id="one_user_more_items_than_k_is_capped",
    ),
    pytest.param(
        # 3 paths but 2 distinct recommended items: counting paths would give 3/k
        [_mk_path(0, 20, link=("item", 10)), _mk_path(0, 20, link=("item", 11)), _mk_path(0, 21, link=("item", 12))],
        {5: 2 / 5, 10: 2 / 10, 20: 2 / 20},
        id="paths_to_the_same_item_count_once",
    ),
    pytest.param(
        # user0: 3 items, user1: 2 items; a global count would give 5/k
        [_mk_path(0, item) for item in (20, 21, 22)] + [_mk_path(1, item) for item in (30, 31)],
        {5: (3 / 5 + 2 / 5) / 2, 10: (3 / 10 + 2 / 10) / 2, 20: (3 / 20 + 2 / 20) / 2},
        id="two_users_mean_over_users",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), FIDELITY_CASES)
def test_fidelity(config, paths, expected):
    """Fidelity@k = mean over users of |explained recommended items| / k, capped at 1."""
    result = _compute(config, "fidelity", {"rec.paths": paths})
    _assert_metric(result, "Fidelity", expected)


# -------------------------------------------------------------------------
# LID
# -------------------------------------------------------------------------

LID_CASES = [
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", 10)) for i in range(5)],
        1 / 5,
        id="all_same_linked_interaction",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", 10 + i)) for i in range(4)],
        1.0,
        id="all_distinct_linked_interactions",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", li)) for i, li in enumerate((10, 10, 11, 11, 12))],
        3 / 5,
        id="mixed_duplicates",
    ),
    pytest.param(
        # user0: 1 unique / 5 paths, user1: 4 unique / 4 paths; a global ratio would give 5/9
        [_mk_path(0, 20 + i, link=("item", 10)) for i in range(5)]
        + [_mk_path(1, 30 + i, link=("item", 20 + i)) for i in range(4)],
        (1 / 5 + 1.0) / 2,
        id="two_users_mean_of_user_ratios",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), LID_CASES)
def test_lid(config, paths, expected):
    """LID(u) = |unique linked interaction ids| / |paths of u|, averaged over users."""
    result = _compute(config, "lid", {"rec.paths": paths})
    _assert_metric(result, "LID", expected)


# -------------------------------------------------------------------------
# SED
# -------------------------------------------------------------------------

SED_CASES = [
    pytest.param(
        # linked items differ, so reading the wrong node would give 1.0
        [_mk_path(0, 20 + i, link=("item", 10 + i), shared=("entity", 999)) for i in range(5)],
        1 / 5,
        id="all_same_shared_entity",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, shared=("entity", 900 + i)) for i in range(4)],
        1.0,
        id="all_distinct_shared_entities",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, shared=("entity", se)) for i, se in enumerate((900, 900, 901, 901, 902))],
        3 / 5,
        id="mixed_duplicates",
    ),
    pytest.param(
        # user0: 1 unique / 3 paths, user1: 2 unique / 2 paths
        [_mk_path(0, 20 + i, link=("item", 10 + i), shared=("entity", 999)) for i in range(3)]
        + [_mk_path(1, 30 + i, shared=("entity", 800 + i)) for i in range(2)],
        (1 / 3 + 1.0) / 2,
        id="two_users_mean_of_user_ratios",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), SED_CASES)
def test_sed(config, paths, expected):
    """SED(u) = |unique shared entity ids| / |paths of u|, averaged over users."""
    result = _compute(config, "sed", {"rec.paths": paths})
    _assert_metric(result, "SED", expected)


# -------------------------------------------------------------------------
# LITD
# -------------------------------------------------------------------------

LITD_CASES = [
    pytest.param(
        # ids differ but the type is always "item": counting ids would give 1.0
        [_mk_path(0, 20 + i, link=("item", 10 + i)) for i in range(5)],
        1 / 5,
        id="same_type_many_ids",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=(t, 10 + i)) for i, t in enumerate(("item", "entity", "user", "brand"))],
        1.0,
        id="all_distinct_types",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=(t, 10 + i)) for i, t in enumerate(("item", "item", "entity", "entity", "brand"))],
        3 / 5,
        id="mixed_duplicate_types",
    ),
    pytest.param(
        # user0: 1 type / 3 paths, user1: 2 types / 2 paths
        [_mk_path(0, 20 + i, link=("item", 10 + i)) for i in range(3)]
        + [_mk_path(1, 30 + i, link=(t, 20 + i)) for i, t in enumerate(("item", "entity"))],
        (1 / 3 + 1.0) / 2,
        id="two_users_mean_of_user_ratios",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), LITD_CASES)
def test_litd(config, paths, expected):
    """LITD(u) = |unique linked interaction types| / |paths of u|, averaged over users."""
    result = _compute(config, "litd", {"rec.paths": paths})
    _assert_metric(result, "LITD", expected)


# -------------------------------------------------------------------------
# SETD
# -------------------------------------------------------------------------

SETD_CASES = [
    pytest.param(
        # ids differ but the type is always "entity": counting ids would give 1.0
        [_mk_path(0, 20 + i, shared=("entity", 900 + i)) for i in range(5)],
        1 / 5,
        id="same_type_many_ids",
    ),
    pytest.param(
        # the last node is always "item", so reading path[-1] instead of path[-2] would give 1/4
        [_mk_path(0, 20 + i, shared=(t, 900 + i)) for i, t in enumerate(("entity", "brand", "genre", "tag"))],
        1.0,
        id="all_distinct_types",
    ),
    pytest.param(
        [
            _mk_path(0, 20 + i, shared=(t, 900 + i))
            for i, t in enumerate(("entity", "entity", "brand", "brand", "genre"))
        ],
        3 / 5,
        id="mixed_duplicate_types",
    ),
    pytest.param(
        # user0: 1 type / 3 paths, user1: 2 types / 2 paths
        [_mk_path(0, 20 + i, shared=("entity", 900 + i)) for i in range(3)]
        + [_mk_path(1, 30 + i, shared=(t, 800 + i)) for i, t in enumerate(("entity", "brand"))],
        (1 / 3 + 1.0) / 2,
        id="two_users_mean_of_user_ratios",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), SETD_CASES)
def test_setd(config, paths, expected):
    """SETD(u) = |unique shared entity types| / |paths of u|, averaged over users."""
    result = _compute(config, "setd", {"rec.paths": paths})
    _assert_metric(result, "SETD", expected)


# -------------------------------------------------------------------------
# PPT
# -------------------------------------------------------------------------

PPT_RID2RELATION = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
PPT_MAX_PATH_LENGTH = 4

PPT_CASES = [
    pytest.param(
        # same relation sequence, different node ids: patterns built from ids would give 1.0
        [_mk_path(0, 20 + i, link=("item", 10 + i)) for i in range(4)],
        1 / 4,
        id="same_pattern_different_node_ids",
    ),
    pytest.param(
        # denominator is min(2, 4) = 2; dividing by max_path_length would give 0.5
        [_mk_path(0, 20, rel_ids=(0, 1, 2, 3)), _mk_path(0, 21, rel_ids=(0, 4, 2, 3))],
        1.0,
        id="fewer_paths_than_max_length",
    ),
    pytest.param(
        # 6 distinct patterns / min(6, 4) = 1.5, capped
        [
            _mk_path(0, 20 + i, rel_ids=rel_ids)
            for i, rel_ids in enumerate(
                ((0, 1, 2, 3), (0, 4, 2, 3), (0, 1, 5, 3), (0, 1, 2, 6), (0, 4, 5, 3), (0, 4, 2, 6))
            )
        ],
        1.0,
        id="more_distinct_patterns_than_max_length_is_capped",
    ),
    pytest.param(
        # user0: 1 pattern / min(4, 4), user1: 2 patterns / min(2, 4)
        [_mk_path(0, 20 + i) for i in range(4)]
        + [_mk_path(1, 30, rel_ids=(0, 1, 2, 3)), _mk_path(1, 31, rel_ids=(0, 4, 2, 3))],
        (1 / 4 + 1.0) / 2,
        id="two_users_mean_of_user_values",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), PPT_CASES)
def test_ppt(config, paths, expected):
    """PPT(u) = min(|unique relation patterns of path[1:]| / min(|paths of u|, max_path_length), 1), averaged."""
    store = {
        "rec.paths": paths,
        "data.max_path_length": PPT_MAX_PATH_LENGTH,
        "data.rid2relation": PPT_RID2RELATION,
    }
    result = _compute(config, "ppt", store)
    _assert_metric(result, "PPT", expected)


# -------------------------------------------------------------------------
# PTC
# -------------------------------------------------------------------------

MAX_PATH_TYPE = [0, 1, 2, 3]

PTC_CASES = [
    pytest.param(
        [_typed_path(0, 20 + i, t) for i, t in enumerate((0, 0, 0, 0))],
        0.0,
        id="all_same_type",
    ),
    pytest.param(
        [_typed_path(0, 20 + i, t) for i, t in enumerate((0, 1, 2, 3))],
        1.0,
        id="all_distinct_types",
    ),
    pytest.param(
        # counts {0: 3, 1: 2}: 1 - (3*2 + 2*1) / (5*4)
        [_typed_path(0, 20 + i, t) for i, t in enumerate((0, 0, 0, 1, 1))],
        0.6,
        id="mixed_distribution",
    ),
    pytest.param(
        # N * (N - 1) = 0 is reported as 0
        [_typed_path(0, 20, 0)],
        0.0,
        id="single_path_is_zero",
    ),
    pytest.param(
        # counts {0: 2, 1: 2}: 1 - (2 + 2) / 12; treating "self_loop" as a type would give 1 - 2/12
        [
            _typed_path(0, 20, 0),
            _typed_path(0, 21, 0, self_loop=True),
            _typed_path(0, 22, 1),
            _typed_path(0, 23, 1, self_loop=True),
        ],
        2 / 3,
        id="self_loop_type_from_penultimate",
    ),
    pytest.param(
        # user0: all same type -> 0.0, user1: counts {0: 3, 1: 2} -> 0.6
        [_typed_path(0, 20 + i, 0) for i in range(4)]
        + [_typed_path(1, 30 + i, t) for i, t in enumerate((0, 0, 0, 1, 1))],
        0.3,
        id="two_users_mean_of_user_values",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), PTC_CASES)
def test_ptc(config, paths, expected):
    """PTC(u) = 1 - sum_t n_t (n_t - 1) / (N (N - 1)), averaged over users."""
    result = _compute(config, "ptc", {"rec.paths": paths, "data.max_path_type": MAX_PATH_TYPE})
    _assert_metric(result, "PTC", expected)


# -------------------------------------------------------------------------
# PTD
# -------------------------------------------------------------------------

PTD_CASES = [
    pytest.param(
        # denominator min(5, 4) = 4; dividing by the number of paths would give 1/5
        [_typed_path(0, 20 + i, 0) for i in range(5)],
        1 / 4,
        id="all_same_type_more_paths_than_types",
    ),
    pytest.param(
        # denominator min(2, 4) = 2; dividing by the number of types would give 1/4
        [_typed_path(0, 20 + i, 0) for i in range(2)],
        1 / 2,
        id="all_same_type_fewer_paths_than_types",
    ),
    pytest.param(
        [_typed_path(0, 20 + i, t) for i, t in enumerate((0, 1, 2, 3))],
        1.0,
        id="all_distinct_types",
    ),
    pytest.param(
        # types {0, 1}; treating "self_loop" as a type would give 3/4
        [
            _typed_path(0, 20, 0),
            _typed_path(0, 21, 0, self_loop=True),
            _typed_path(0, 22, 1),
            _typed_path(0, 23, 1, self_loop=True),
        ],
        2 / 4,
        id="self_loop_type_from_penultimate",
    ),
    pytest.param(
        # user0: 1 type / min(5, 4), user1: 4 types / min(4, 4)
        [_typed_path(0, 20 + i, 0) for i in range(5)]
        + [_typed_path(1, 30 + i, t) for i, t in enumerate((0, 1, 2, 3))],
        (1 / 4 + 1.0) / 2,
        id="two_users_mean_of_user_values",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), PTD_CASES)
def test_ptd(config, paths, expected):
    """PTD(u) = |unique path types| / min(|paths of u|, |max_path_type|), averaged over users."""
    result = _compute(config, "ptd", {"rec.paths": paths, "data.max_path_type": MAX_PATH_TYPE})
    _assert_metric(result, "PTD", expected)


# -------------------------------------------------------------------------
# LIR
# -------------------------------------------------------------------------

LIR_NUM_ITEMS = 6
# Exactly two interactions per user, so the normalized EMA is 0 for the older one and 1 for the recent one.
LIR_TIMESTAMPS = np.array(
    [
        [0, 100, 0, 200, 0, 0],  # user0: item1 older, item3 recent
        [50, 0, 150, 0, 0, 0],  # user1: item0 older, item2 recent
    ],
    dtype=np.float32,
)

LIR_CASES = [
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", 1)) for i in range(3)],
        0.0,
        id="only_older_linked_item",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", 3)) for i in range(2)],
        1.0,
        id="only_recent_linked_item",
    ),
    pytest.param(
        [_mk_path(0, 20, link=("item", 1)), _mk_path(0, 21, link=("item", 3))],
        0.5,
        id="mixed_older_and_recent",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, link=("item", 4)) for i in range(2)],
        0.0,
        id="linked_item_not_in_history",
    ),
    pytest.param(
        [_mk_path(0, 20, link=("entity", 3))],
        1.0,
        id="entity_node_with_item_id_is_accepted",
    ),
    pytest.param(
        # id 6 >= num_items is not an item: it must be skipped rather than averaged in
        [_mk_path(0, 20, link=("entity", 3)), _mk_path(0, 21, link=("entity", LIR_NUM_ITEMS))],
        1.0,
        id="entity_node_with_non_item_id_is_skipped",
    ),
    pytest.param(
        # (1 + 1 + 1 + 0) / 4 over paths; averaging per user first would give 0.5
        [_mk_path(0, 20 + i, link=("item", 3)) for i in range(3)] + [_mk_path(1, 30, link=("item", 0))],
        0.75,
        id="two_users_mean_over_paths",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), LIR_CASES)
def test_lir(config, paths, expected):
    """LIR = mean over paths of the normalized EMA recency of the linked interaction."""
    store = {"rec.paths": paths, "data.timestamp": LIR_TIMESTAMPS, "data.num_items": LIR_NUM_ITEMS}
    result = _compute(config, "lir", store)
    _assert_metric(result, "lir", expected)


# -------------------------------------------------------------------------
# SEP
# -------------------------------------------------------------------------

# Two nodes per type, so the normalized EMA is 0 for the less popular one and 1 for the more popular one.
SEP_NODE_DEGREE = {
    "entity": {900: 1, 901: 2},
    "brand": {800: 5, 801: 10},
}

SEP_CASES = [
    pytest.param(
        [_mk_path(0, 20 + i, shared=("entity", 900)) for i in range(3)],
        0.0,
        id="all_less_popular_entity",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, shared=("entity", se)) for i, se in enumerate((900, 901, 901, 900))],
        0.5,
        id="mixed_popularity_mean_over_paths",
    ),
    pytest.param(
        [_mk_path(0, 20 + i, shared=("item", 901)) for i in range(2)],
        1.0,
        id="item_shared_type_maps_to_entity",
    ),
    pytest.param(
        # normalized per type; a single distribution over degrees [1, 2, 5, 10] would change every value
        [
            _mk_path(0, 20, shared=("brand", 800)),
            _mk_path(0, 21, shared=("brand", 801)),
            _mk_path(0, 22, shared=("entity", 901)),
        ],
        (0.0 + 1.0 + 1.0) / 3,
        id="types_are_normalized_separately",
    ),
]


@pytest.mark.parametrize(("paths", "expected"), SEP_CASES)
def test_sep(config, paths, expected):
    """SEP = mean over paths of the per-type normalized EMA popularity of the shared entity."""
    result = _compute(config, "sep", {"rec.paths": paths, "data.node_degree": SEP_NODE_DEGREE})
    _assert_metric(result, "SEP", expected)
