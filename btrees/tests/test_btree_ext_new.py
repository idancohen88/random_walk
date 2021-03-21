import os
from contextlib import contextmanager
from unittest.mock import patch

from BTrees import OOBTree

import pandas as pd
from build_tree import build_tree
from build_tree.build_tree import ALPHABET, overriding_btree_max_leaf_size
from BTrees.OOBTree import OOBTreePy

from btrees.btee_ext import OOBTreeExt, SAMPLING_TESTS_CSV

MAX_INTERNAL_SIZE = 5
MAX_LEAF_SIZE = 5


def test_oklen_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_olken(sample_size)) == sample_size


def test_ours_height_three_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert (
        len(my_index.sample_distribution_oriented_height_three(sample_size))
        == sample_size
    )


def test_oklen__early_abort_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_olken_early_abort(sample_size)) == sample_size


def _generate_3_height_btree():
    # mocking BTrees/OOBTree.py:276
    # todo: change it to mocking, as it's not safe
    # OOBTreeExt.max_leaf_size = MAX_LEAF_SIZE
    # OOBTreeExt.max_internal_size = MAX_INTERNAL_SIZE
    with overriding_btree_max_leaf_size(max_leaf_size=MAX_LEAF_SIZE), patch.object(
        OOBTreePy, "max_internal_size", MAX_INTERNAL_SIZE
    ):
        my_index = OOBTreeExt()
        my_index.has_key("a")
        for i, c in enumerate(ALPHABET):
            my_index[c] = i

    assert isinstance(
        my_index._data[0].child._data[0].child, my_index._bucket_type
    ), "3 height btree must arrive to bucket after wo steps"

    return my_index


def test_btree_generation__custom_leaf_size():
    prefix_to_percent = {"gggg": 0.53, "": 0.47}
    max_leaf_size = 14
    with overriding_btree_max_leaf_size(max_leaf_size=max_leaf_size):
        my_index = build_tree.generate_btree_index_x_values_with_dist(
            num_of_values=5000, disired_prefix_to_percent_dist=prefix_to_percent
        )

    bucket = my_index._firstbucket
    assert bucket.size <= max_leaf_size
    while bucket._next:
        assert bucket.size <= max_leaf_size
        bucket = bucket._next


def _generate_4_height_btree():
    my_index = OOBTreeExt()
    my_index.max_internal_size = 3
    my_index.max_leaf_size = 3
    my_index.has_key("a")
    for i, c in enumerate(ALPHABET * 3):
        my_index[i] = c

    assert isinstance(
        my_index._data[0].child._data[0].child._data[0].child, my_index._bucket_type
    ), "4 height btree must arrive to bucket after wo steps"
    return my_index


def test_sample_distribution_height_four():
    sample_size = 3
    my_index = _generate_4_height_btree()
    assert (
        len(my_index.sample_distribution_oriented_height_four(sample_size))
        == sample_size
    )


def test_ours_height_four__walk_to_determine_root_coefs():
    my_index = _generate_4_height_btree()
    my_index._fanout_distribution_cache = {}  # todo: change!!
    from collections import Counter

    my_index._cache_hit_counter = Counter()  # todo: change!!
    coef = my_index._first_walk_to_determine_root_coefs()

    assert sum(coef) == 1


def test_persisting_stats():
    if os.path.exists(SAMPLING_TESTS_CSV):
        os.remove(SAMPLING_TESTS_CSV)
    my_index = _generate_4_height_btree()
    my_index.sample_olken(1)
    my_index.sample_distribution_oriented_height_four(1)

    csv = pd.read_csv(SAMPLING_TESTS_CSV)
    assert len(csv) == 2
    expected_columns = {
        "sample_size", "p_value", "ks_stats", "name", "start_time", "sampled_values_counter",
        "running_time", "reject_counter",}
    assert set(csv.columns) == expected_columns



if __name__ == "__main__":
    if os.path.exists(SAMPLING_TESTS_CSV):
        os.remove(SAMPLING_TESTS_CSV)

    test_btree_generation__custom_leaf_size()
    test_oklen_sanity()
    test_oklen__early_abort_sanity()
    test_ours_height_three_sanity()
    test_ours_height_four__walk_to_determine_root_coefs()
    test_sample_distribution_height_four()
    test_persisting_stats()
