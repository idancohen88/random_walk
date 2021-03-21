import os
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

from btrees.btree_sampling_olken import OOBTreeExtOlken
from btrees.btree_sampling_distribution_oriented import OOBTreeExtFanoutOriented

SAMPLING_TESTS_CSV = "sampling_tests.csv"

STATS_TOP_X_TO_SHOW = 10


class OOBTreeExt(OOBTreeExtOlken, OOBTreeExtFanoutOriented):
    def __init__(self):
        super(OOBTreeExt).__init__()
        self._np_samples = []
        self._btree_size = None
        self._real_max_leaf_size = self._get_max_leaf_size_at_init()
        self._real_max_internal_size = self._get_max_internal_size_at_init()
        self._btree_height = None

    def run_all_samples(self, k):
        self.sample_distribution_oriented_height_four(k)
        self.sample_distribution_oriented_height_three(k)
        self.sample_olken(k)
        self.sample_olken_early_abort(k)

    def _get_max_leaf_size_at_init(self):
        # saving the value, even if it's mocked
        return self.max_leaf_size

    def _get_max_internal_size_at_init(self):
        # saving the value, even if it's mocked
        return self.max_internal_size

    def _persist_sampling_stats(self, **kwargs):
        end_time = datetime.now()
        name = kwargs["name"]
        start_time = kwargs["start_time"]
        sampled_tuples = kwargs["sampled_values"]
        sampled_values = [x[1] for x in sampled_tuples]
        sample_size = kwargs["sample_size"]
        reject_counter = kwargs["reject_counter"] if "olken" in name else None

        running_time = (end_time - start_time).seconds

        ks_stats, p_value = self._calculate_ks_test(sampled_values, sample_size)

        self._btree_size = self._btree_size or len(self.values())

        sampled_values_counter = Counter(sampled_values).most_common(
            STATS_TOP_X_TO_SHOW
        )

        sampled_csv = self._append_to_df(
            name=name,
            start_time=start_time,
            sample_size=sample_size,
            reject_counter=reject_counter,
            running_time=running_time,
            ks_stats=ks_stats,
            p_value=p_value,
            sampled_values_counter=sampled_values_counter,
            max_leaf_size=self._real_max_leaf_size,
            max_internal_size=self._real_max_internal_size,
            btree_size=self._btree_size,
            btree_height=self._btree_height or self._get_height()
        )
        return sampled_csv

    def _append_to_df(self, **kwargs):
        samples_df = get_samples_csv()

        samples_df = samples_df.append([kwargs], ignore_index=True)
        samples_df.to_csv(SAMPLING_TESTS_CSV, index=False)
        return samples_df

    def _calculate_ks_test(self, sampled_values, sample_size):
        if len(self._np_samples) == 0 or len(self._np_samples) != sample_size:
            self._np_samples = np.random.choice(self.values(), sample_size)
        return stats.ks_2samp(self._np_samples, sampled_values)

    def _get_random_step_and_value_from_bucket(self, bucket):
        next_random_step = np.random.randint(low=0, high=bucket.size)

        value_in_leaf = bucket.items()[next_random_step]
        return next_random_step, value_in_leaf

    def _get_height(self):
        h = 1
        node = self._data[0].child
        while not isinstance(node, self._bucket_type):
            node = node._data[0].child
            h += 1
        return h + 1

def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob


def get_samples_csv():
    if os.path.isfile(SAMPLING_TESTS_CSV):
        return pd.read_csv(SAMPLING_TESTS_CSV)
    return pd.DataFrame()

