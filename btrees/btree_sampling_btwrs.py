from datetime import datetime

import numpy as np
from btrees.btree_base import OOBTreeBase


class OOBTreeExtBtwrs(OOBTreeBase):  # BTree based Weighted Random Sampling
    def sample_btwrs(self, k):
        k = self._min_between_k_and_btree_size(k)
        print(f"sampling {k}\{self._btree_size} using btwrs")
        start_time = datetime.now()
        sampled_values = []
        sampled_path = []
        reject_counter = 0
        while len(sampled_values) < k:
            value, acc_rej_prob, path = self._btwrs_walk_toward_bucket(node=self)

            if path in sampled_path:
                continue  # todo: revisited

            if _accept_reject_test_pass(acc_rej_prob):
                sampled_values.append(value)
                sampled_path.append(path)
                continue

            reject_counter += 1

        self._persist_sampling_stats(
            reject_counter=reject_counter,
            start_time=start_time,
            sampled_values=sampled_values,
            name="btwrs",
            sample_size=k,
        )
        return sampled_values

    def _btwrs_walk_toward_bucket(self, node):
        walking_path = []
        current_node = node
        acc_rej_test_max_fan_out = self.max_internal_size
        acc_rej_test_acceptance_prob = node.size / acc_rej_test_max_fan_out

        while not isinstance(current_node, self._bucket_type):
            next_random_step = np.random.randint(low=0, high=current_node.size)
            current_node = current_node._data[next_random_step].child
            acc_rej_test_acceptance_prob *= current_node.size / acc_rej_test_max_fan_out
            walking_path.append(next_random_step)

        next_random_step = np.random.randint(low=0, high=current_node.size)
        value_in_leaf = current_node.items()[next_random_step]

        walking_path.append(next_random_step)
        return value_in_leaf, acc_rej_test_acceptance_prob, walking_path


def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob
