from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

import numpy as np
from BTrees.OOBTree import OOBTreePy

from btrees.btee_ext import OOBTreeExt

CHUNKS_SIZE = 10000
KEY_LENGTH = 8
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def generate_btree_index_x_values_with_dist(
    num_of_values, disired_prefix_to_percent_dist, my_index=None
):
    print("start time %s" % datetime.now())
    my_index = my_index if my_index is not None else OOBTreeExt()
    for prefix, amount_percent in disired_prefix_to_percent_dist.items():
        amount = int(num_of_values * amount_percent)
        my_index = insert_to_index_random(my_index, amount, prefix)

    print("finish at %s" % datetime.now())
    return my_index


def generate_btree_index_x_values_with_dist_and_custom_leaf_size(
    num_of_values, disired_prefix_to_percent_dist, my_index=None, leaf_size=None
):
    assert leaf_size
    with overriding_btree_max_leaf_size(leaf_size):
        my_index = generate_btree_index_x_values_with_dist(
            num_of_values, disired_prefix_to_percent_dist, my_index
        )

    _validate_leaf_size(my_index, leaf_size)
    return my_index


def _validate_leaf_size(my_index, max_leaf_size):
    bucket = my_index._firstbucket
    assert bucket.size <= max_leaf_size
    while bucket._next:
        assert bucket.size <= max_leaf_size
        bucket = bucket._next


def insert_to_index_random(my_index, amount, prefix=""):
    amount_in_iteration = min(CHUNKS_SIZE, amount)
    print(
        "generating %s values, chunk of %s, with prefix='%s'"
        % (amount, amount_in_iteration, prefix)
    )

    proceed = 0
    for i in range(0, amount, amount_in_iteration):
        alphabet = list(ALPHABET)
        np_alphabet = np.array(alphabet)
        np_codes = np.random.choice(np_alphabet, [amount_in_iteration, KEY_LENGTH])
        data_to_insert = {
            prefix + "".join(np_codes[i]): prefix
            for i in range(len(np_codes))
        }
        my_index.update(data_to_insert)

        proceed += amount_in_iteration
        if (proceed % 150000) == 0:
            print("done generating %s values" % (proceed))
    return my_index


@contextmanager
def overriding_btree_max_leaf_size(max_leaf_size):
    try:
        print('warning!!! mocking max_leaf_size!!!')
        with patch.object(OOBTreePy, "max_leaf_size", max_leaf_size):
            yield
    finally:
        print('warning!!! mocking max_leaf_size has over!!')

