import os
from unittest.mock import patch
import matplotlib.pyplot as plt


import pandas as pd
import pytest
from numpy import random

from btrees import utils
from btrees.btree_base import SAMPLING_METHODS, DUMMIES_SAMPLING_METHODS
from build_tree import build_tree
from build_tree.build_tree import ALPHABET, overriding_btree_max_leaf_size, generate_zipf_dist, \
    generate_zipf_dist_in_random_order
from BTrees.OOBTree import OOBTreePy

from btrees.btree_ext import OOBTreeExt
from btrees.common import SAMPLING_TESTS_CSV

MAX_INTERNAL_SIZE = 5
MAX_LEAF_SIZE = 5


CSV_FIELDS = {"sample_size", "p_value", "ks_stats", "name", "start_time", "sampled_values_counter",
        "running_time", "reject_counter", "max_leaf_size", "max_internal_size", "btree_size",
        "btree_height", "distinct_values_error", "skew_factor", "domain_size" , "data_generation_method", "btree_id", "dist_equality_score"}
MANDATORY_FIELDS = CSV_FIELDS - {"p_value", "reject_counter", "skew_factor", "domain_size"}


@pytest.fixture(autouse=True)
def remove_csv_file():
    if os.path.exists(SAMPLING_TESTS_CSV):
        os.remove(SAMPLING_TESTS_CSV)

def test_oklen_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_olken(sample_size)) == sample_size

def test_btwrs_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_btwrs(sample_size)) == sample_size

def test_monkey_sample():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_monkey(sample_size)) == sample_size

def test_naive_random_walk_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_naive_random_walk(sample_size)) == sample_size

    sampled_path_key_tuples = list(my_index.sampled_paths.keys())[0]
    sample_name, sample_size = sampled_path_key_tuples
    relevant_sampled_path = my_index.sampled_paths[sampled_path_key_tuples][0]
    assert len(relevant_sampled_path) == sample_size

def test_all_sampling_methods_write_to_csv_with_all_metadata__also_dummies():
    my_index = _generate_3_height_btree()
    my_index.run_all_samples(k=1, iterations=1, run_also_dummies=True)
    csv = pd.read_csv(SAMPLING_TESTS_CSV)

    expected_sampling_methods = len(SAMPLING_METHODS) + len(DUMMIES_SAMPLING_METHODS)

    assert len(csv) == expected_sampling_methods

    assert all([len(csv[field].notnull()) == expected_sampling_methods for field in MANDATORY_FIELDS])



def test_all_sampling_methods_write_to_csv_with_all_metadata():
    my_index = _generate_3_height_btree()
    my_index.run_all_samples(k=1, iterations=1)
    csv = pd.read_csv(SAMPLING_TESTS_CSV)

    assert len(csv) == len(SAMPLING_METHODS)

    assert all([len(csv[field].notnull()) == len(SAMPLING_METHODS) for field in MANDATORY_FIELDS])


def test_btwrs_vs_olken_higher_prob():
    my_index = _generate_3_height_btree()
    with patch.object(random, 'randint', return_value=0):
        _, olken_prob, olken_path = my_index._olken_walk_toward_bucket(node=my_index)
        _, btwrs_prob, btwrs_path = my_index._btwrs_walk_toward_bucket(node=my_index)

    assert olken_path == btwrs_path == [0, 0, 0]

    # assert btwrs_prob > olken_prob

def test_ours_height_three_sanity():
    sample_size = 3
    my_index = _generate_3_height_btree()
    sampled_dist_oritented = my_index.sample_distribution_oriented_height_three(sample_size)
    assert len(sampled_dist_oritented) == sample_size

    sampled_path_key_tuples = list(my_index.sampled_paths.keys())[0]
    sample_name, sample_size = sampled_path_key_tuples
    relevant_sampled_path = my_index.sampled_paths[sampled_path_key_tuples][0]
    assert len(relevant_sampled_path) == sample_size

def test_dataframe_to_histogram():
    df_dataset = {"sampled_values_counter": "[('', 247), ('gggg', 117), ('hhhh', 77), ('mmmm', 53), ('rrrr', 6)]",
                  "sample_size":1, "btree_size":1, "name":"olken", "btree_height":3, "max_leaf_size":3}
    df = pd.DataFrame([df_dataset] * 4 )

    with patch.object(plt, 'show'):
        utils.dataframe_to_histogram(df)

def test_olken__early_abort_sanity():
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
            my_index[i] = c

    assert isinstance(
        my_index._data[0].child._data[0].child, my_index._bucket_type
    ), "3 height btree must arrive to bucket after wo steps"

    my_index._data_generation_method = 'test'
    return my_index

def test_btree_generation__custom_leaf_size_zipf_dist():
    max_leaf_size = 10

    wide_leaf_index = build_tree.generate_zipf_dist_custom_leaf(
        num_of_values=50, domain_size=50, skew_factor=0, leaf_size=max_leaf_size)
    assert wide_leaf_index._real_max_leaf_size == max_leaf_size
    assert all(bucket.size <= max_leaf_size for bucket in _iter_buckets(wide_leaf_index))



def test_btree_generation__custom_leaf_size():
    prefix_to_percent = {"gggg": 0.53, "": 0.47}
    max_leaf_size = 14
    with overriding_btree_max_leaf_size(max_leaf_size=max_leaf_size):
        my_index = build_tree.generate_btree_index_x_values_with_dist(
            num_of_values=5000, disired_prefix_to_percent_dist=prefix_to_percent
        )

    assert all(bucket.size <= max_leaf_size for bucket in _iter_buckets(my_index))
    bucket = my_index._firstbucket
    assert bucket.size <= max_leaf_size
    while bucket._next:
        assert bucket.size <= max_leaf_size
        bucket = bucket._next

def _iter_buckets(index):
    bucket = index._firstbucket
    yield bucket
    while bucket._next:
        bucket = bucket._next
        yield bucket

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
    my_index._data_generation_method = 'test'
    return my_index


def test_sample_distribution_height_four():
    sample_size = 3
    my_index = _generate_4_height_btree()
    my_index._data_generation_method = 'test'
    sampled_values = my_index.sample_distribution_oriented_height_four(sample_size)
    assert len(sampled_values) == sample_size

    sampled_path_key_tuples = list(my_index.sampled_paths.keys())[0]
    sample_name, sample_size = sampled_path_key_tuples
    relevant_sampled_path = my_index.sampled_paths[sampled_path_key_tuples][0]
    assert len(relevant_sampled_path) == sample_size


def test_ours_height_four__walk_to_determine_root_coefs():
    my_index = _generate_4_height_btree()
    coef = my_index._first_walk_to_determine_root_coefs()

    assert sum(coef) == 1


def test_persisting_stats():
    my_index = _generate_4_height_btree()
    my_index.sample_olken(1)
    my_index.sample_distribution_oriented_height_four(1)

    csv = pd.read_csv(SAMPLING_TESTS_CSV)
    assert len(csv) == 2
    assert set(csv.columns) == CSV_FIELDS

def test_get_height():
    my_index = _generate_4_height_btree()
    assert my_index._get_height() == 4


def test_distinct_values_estimator():
    my_index = _generate_3_height_btree()
    my_index.update({key:value for key, value in enumerate(list(iter('aaabbbccddddd')) * 20, 100)})
    # continue from here, check alg validation
    sample_size = 30
    sampled_tuples = my_index.sample_distribution_oriented_height_four(sample_size)
    sampled_values = [x[1] for x in sampled_tuples]
    distinct_values_metric = my_index._distinct_values_error_metric(sampled_values, sample_size)
    assert distinct_values_metric > 0

    distinct_values_metric_on_entire_data = my_index._distinct_values_error_metric(my_index.values(), sample_size)
    assert distinct_values_metric > distinct_values_metric_on_entire_data

def test_run_all_samples__sanity():
    my_index = _generate_4_height_btree()
    my_index._data_generation_method = 'test'
    my_index.run_all_samples(1,1)
    my_index._data_generation_method = 'test'

def test_generate_zipf_dist_random_order__sanity():
    my_index_uniform = generate_zipf_dist_in_random_order(num_of_values=50, domain_size=50, skew_factor=0)
    my_index_skewed = generate_zipf_dist_in_random_order(num_of_values=50, domain_size=50, skew_factor=0.9)
    assert set(my_index_uniform.values()) > set(my_index_skewed.values())


def test_generate_zipf_dist__sanity():
    my_index_uniform = generate_zipf_dist(num_of_values=50, domain_size=50, skew_factor=0)
    my_index_skewed = generate_zipf_dist(num_of_values=50, domain_size=50, skew_factor=0.9)
    assert set(my_index_uniform.values()) > set(my_index_skewed.values())


def test_all_samples_protected_from_big_k_size():
    my_index = OOBTreeExt()
    my_index._data_generation_method = 'test'
    my_index.update({'a':1})
    assert my_index.size == 1
    my_index.run_all_samples(k=2)
    my_index._data_generation_method = 'test'
    assert True, 'otherwise, never finish'
