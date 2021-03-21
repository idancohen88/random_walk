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
        self._np_samples = None
        self._btree_size = None

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
        )
        return sampled_csv

    def _append_to_df(self, **kwargs):
        if os.path.isfile(SAMPLING_TESTS_CSV):
            samples_df = pd.read_csv(SAMPLING_TESTS_CSV)
        else:
            samples_df = pd.DataFrame()

        samples_df = samples_df.append([kwargs], ignore_index=True)
        samples_df.to_csv(SAMPLING_TESTS_CSV, index=False)
        return samples_df

    def _calculate_ks_test(self, sampled_values, sample_size):
        if not self._np_samples or len(self._np_samples) != sample_size:
            self._np_samples = np.random.choice(self.values(), sample_size)
        return stats.ks_2samp(self._np_samples, sampled_values)

    def _get_random_step_and_value_from_bucket(self, bucket):
        next_random_step = np.random.randint(low=0, high=bucket.size)

        value_in_leaf = bucket.items()[next_random_step]
        return next_random_step, value_in_leaf


def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob
