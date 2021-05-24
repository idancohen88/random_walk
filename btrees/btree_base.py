from collections import Counter
from datetime import datetime
from itertools import groupby
from math import sqrt
from operator import itemgetter
import numpy as np
from BTrees.OOBTree import OOBTreePy
from scipy import stats

from btrees.common import SAMPLING_TESTS_CSV
from btrees.utils import get_samples_csv

STATS_TOP_X_TO_SHOW = 10

SAMPLING_METHODS = [
    "sample_distribution_oriented_height_four",
    "sample_distribution_oriented_height_three",
    "sample_olken",
    "sample_olken_early_abort",
    "sample_btwrs",
]

DUMMIES_SAMPLING_METHODS = ["sample_monkey", "sample_numpy", "sample_naive_random_walk"]


class OOBTreeBase(OOBTreePy):
    def __init__(self):
        super(OOBTreeBase).__init__()
        self._np_samples = []
        self._btree_size_value = None
        self._real_max_leaf_size = self._get_max_leaf_size_at_init()
        self._real_max_internal_size = self._get_max_internal_size_at_init()
        self._btree_height_value = None
        self._num_distinct_values = None
        self._skew_factor = None
        self._domain_size = None
        self._data_generation_method = None
        self.btree_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")


    @property
    def _btree_size(self):
        self._btree_size_value = self._btree_size_value or len(self.values())
        return self._btree_size_value

    @property
    def _btree_height(self):
        self._btree_height_value = self._btree_height_value or self._get_height()
        return self._btree_height_value

    def run_all_samples(self, k, iterations=1, run_also_dummies=False):
        if isinstance(k, int):
            k = [k]

        samplings_methods = SAMPLING_METHODS + (DUMMIES_SAMPLING_METHODS if run_also_dummies else [])

        for sample_size in k:
            for i in range(iterations):
                print(f"{datetime.now()} - sample size {sample_size} iteration {i}")
                for sampling_method in samplings_methods:
                    print(f"{datetime.now()} - sampling {sample_size}\{self._btree_size} using {sampling_method}")
                    method_callable = getattr(self, sampling_method)
                    method_callable(sample_size)

    def _get_max_leaf_size_at_init(self):
        # saving the value, even if it's mocked
        return self.max_leaf_size

    def _get_max_internal_size_at_init(self):
        # saving the value, even if it's mocked
        return self.max_internal_size

    def _persist_sampling_stats(self, **kwargs):
        assert self._data_generation_method, "must define _data_generation_method"
        end_time = datetime.now()
        name = kwargs["name"]
        start_time = kwargs["start_time"]
        sampled_tuples = kwargs["sampled_values"]
        sampled_values = [x[1] for x in sampled_tuples]
        sample_size = kwargs["sample_size"]
        reject_counter = kwargs.get("reject_counter")

        running_time = (end_time - start_time).seconds

        ks_stats, p_value = self._calculate_ks_test(sampled_values, sample_size)

        sampled_values_counter = Counter(sampled_values).most_common(
            STATS_TOP_X_TO_SHOW
        )

        distinct_values_error_metric = self._distinct_values_error_metric(
            sampled_values, sample_size
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
            btree_height=self._btree_height,
            distinct_values_error=distinct_values_error_metric,
            skew_factor=self._skew_factor,
            domain_size=self._domain_size,
            data_generation_method=self._data_generation_method,
            btree_id=self.btree_id
        )

        self._clean_counters()
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

    def _distinct_values_error_metric(self, sampled_values, sample_size):
        if not sampled_values:
            return 0

        distinct_values_estimator = self._distinct_values_estimator(
            sample_size, sampled_values
        )
        self._num_distinct_values = self._num_distinct_values or len(set(self.values()))
        rel_error = (
            self._num_distinct_values - distinct_values_estimator
        ) / self._btree_size

        return rel_error

    def _distinct_values_estimator(self, sample_size, sampled_values):
        group_size_to_number_of_groups = self._calculate_group_size_to_number_of_groups(
            sampled_values
        )

        f1 = group_size_to_number_of_groups.get(1, 1)
        group_size_to_number_of_groups.pop(1, None)
        all_other_f_sums = sum(group_size_to_number_of_groups.values())
        estimator = sqrt(self._btree_size / sample_size) * f1 + all_other_f_sums
        return estimator

    def _calculate_group_size_to_number_of_groups(self, sampled_values):
        count_values = Counter(sampled_values)
        count_values_ordered = sorted(count_values.items(), key=itemgetter(1))
        group_size_to_number_of_groups = {}
        for _, group in groupby(count_values_ordered, itemgetter(1)):
            group = list(group)
            groups_of_size = group[0][1]
            number_of_groups = len(group)
            group_size_to_number_of_groups[groups_of_size] = number_of_groups
        return group_size_to_number_of_groups

    def set_skew_factor(self, skew_factor):
        self._skew_factor = skew_factor

    def set_data_generation_method(self, method):
        self._data_generation_method = method

    def set_domain_size(self, domain_size):
        self._domain_size = domain_size

    def _clean_counters(self):
        self._fanout_distribution_cache = {}
        self._cache_hit_counter = Counter()

    def _min_between_k_and_btree_size(self, k):
        return min(k, self._btree_size)


def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob
