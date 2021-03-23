from datetime import datetime

import numpy as np
from btrees.btee_base import OOBTreeBase


class OOBTreeExtOlken(OOBTreeBase):
    def sample_olken_early_abort(self, k):
        start_time = datetime.now()
        sampled_values = []
        sampled_paths = []
        reject_counter = 0
        while len(sampled_values) < k:
            value, path = self._olken_walk_toward_bucket_early_abort(node=self)

            if value is None:
                reject_counter += 1
                continue

            if path in sampled_paths:
                continue  # todo: revisited

            sampled_values.append(value)
            sampled_paths.append(path)

        self._persist_sampling_stats(reject_counter=reject_counter, start_time=start_time, sampled_values=sampled_values,
                                     name='olken_early_abort', sample_size=k)

        return sampled_values

    def sample_olken(self, k):
        start_time = datetime.now()
        sampled_values = []
        sampled_path = []
        reject_counter = 0
        while len(sampled_values) < k:
            value, acc_rej_prob, path = self._olken_walk_toward_bucket(node=self)

            if path in sampled_path:
                continue  # todo: revisited

            if _accept_reject_test_pass(acc_rej_prob):
                sampled_values.append(value)
                sampled_path.append(path)
                continue

            reject_counter += 1

        self._persist_sampling_stats(reject_counter=reject_counter, start_time=start_time, sampled_values=sampled_values,
                                     name='olken', sample_size=k)
        return sampled_values

    def _olken_walk_toward_bucket(self, node):
        walking_path = []
        current_node = node
        acc_rej_test_acceptance_prob = 1
        acc_rej_test_max_fan_out = self.max_internal_size

        while not isinstance(current_node, self._bucket_type):
            next_random_step = np.random.randint(low=0, high=current_node.size)
            current_node = current_node._data[next_random_step].child
            acc_rej_test_acceptance_prob *= current_node.size / acc_rej_test_max_fan_out
            walking_path.append(next_random_step)

        next_random_step = np.random.randint(low=0, high=current_node.size)
        value_in_leaf = current_node.items()[next_random_step]

        # acc_rej_test_acceptance_prob *= 1 / current_node.size # todo: not sure that need test in leaf level

        walking_path.append(next_random_step)
        return value_in_leaf, acc_rej_test_acceptance_prob, walking_path

    def _olken_walk_toward_bucket_early_abort(self, node):
        walking_path = []
        current_node = node

        while not isinstance(current_node, self._bucket_type):
            next_random_step = np.random.randint(low=0, high=current_node.size)
            current_node = current_node._data[next_random_step].child
            if not self._early_abort_should_continue(current_node):
                return None, None

            walking_path.append(next_random_step)


        next_random_step, value_in_leaf = self._get_random_step_and_value_from_bucket(bucket=current_node)
        walking_path.append(next_random_step)
        return value_in_leaf, walking_path

    def _early_abort_should_continue(self, node):
        return _accept_reject_test_pass((node.size / self.max_internal_size))


def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob