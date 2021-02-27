import os
os.environ["PURE_PYTHON"] = "True"

from BTrees.OOBTree import OOBTree as _OOBTree
from collections import Counter
import numpy as np


_debug_random_sampling = []


class WALKING_TECHNIQUE:
    RANDOM = 'random'
    RANDOM_WITH_TEST = 'random_with_test'
    RANDOM_WITH_TEST_EARLY_ABORT = 'random_with_test_early_abort'
    DISTRIBUTION_ORIENTED = 'distribution_oriented'


class OOBTreeExt(_OOBTree):

    def __init__(self):
        super(OOBTreeExt, self).__init__()
        self.walking_path_to_fanout_distribution = {}

    def _get_height(self):
        h = 1
        node = self._data[0].child
        while not isinstance(node, self._bucket_type):
            node = node._data[0].child
            h += 1
        return h + 1

    def random_sampling(self, k, how_to_walk):
        self.height = self._get_height()
        self.walking_path_to_fanout_distribution = {}
        all_accept_reject_measures = {
            'accept': [],
            'reject': [],
            'revisited_paths': Counter()
        }

        k = min(len(self), k)
        sampled_values = []
        all_walking_paths_set = set()
        while len(sampled_values) < k:
            sampled_value, walking_path, acc_rej_test_acceptance_prob = \
                self._get_value_and_path_by_random_walk_from_node(node=self,
                    how_to_walk=how_to_walk)

            if how_to_walk == WALKING_TECHNIQUE.RANDOM_WITH_TEST_EARLY_ABORT and sampled_value is None:
                accept_reject_measures = {
                    'path': walking_path,
                    'value': sampled_value,
                    'acceptance_prob': acc_rej_test_acceptance_prob
                }
                all_accept_reject_measures['reject'].append(accept_reject_measures)
                continue


            if _this_value_was_sampled_already(walking_path, all_walking_paths_set):
                all_accept_reject_measures['revisited_paths'][str(walking_path)] += 1
                continue

            accept_reject_measures = {
                'path': walking_path,
                'value': sampled_value,
                'acceptance_prob': acc_rej_test_acceptance_prob
            }

            if how_to_walk == WALKING_TECHNIQUE.RANDOM_WITH_TEST and not _accept_reject_test_pass(
                    acc_rej_test_acceptance_prob):
                all_accept_reject_measures['reject'].append(accept_reject_measures)
                continue

            all_accept_reject_measures['accept'].append(accept_reject_measures)

            all_walking_paths_set.add(str(walking_path))
            sampled_values.append(sampled_value)

        add_to_debug_global(locals())

        return sampled_values

    def _get_value_and_path_by_random_walk_from_node(self, node, how_to_walk):
        walking_path = []
        current_node = node
        acc_rej_test_acceptance_prob = 1
        acc_rej_test_max_fan_out = None

        while not isinstance(current_node, self._bucket_type):
            if how_to_walk in (WALKING_TECHNIQUE.RANDOM_WITH_TEST, WALKING_TECHNIQUE.RANDOM_WITH_TEST_EARLY_ABORT):
                acc_rej_test_max_fan_out = self.max_internal_size
                next_random_step = np.random.randint(low=0, high=current_node.size)
            elif how_to_walk == WALKING_TECHNIQUE.DISTRIBUTION_ORIENTED:
                next_random_step = self._random_next_move_respect_fanout_prob(current_node, walking_path)
            else:
                assert how_to_walk == WALKING_TECHNIQUE.RANDOM
                next_random_step = np.random.randint(low=0, high=current_node.size)

            current_node = current_node._data[next_random_step].child

            if how_to_walk == WALKING_TECHNIQUE.RANDOM_WITH_TEST:
                acc_rej_test_acceptance_prob *= (current_node.size / acc_rej_test_max_fan_out)
            if how_to_walk == WALKING_TECHNIQUE.RANDOM_WITH_TEST_EARLY_ABORT:
                should_continue = _accept_reject_test_pass((current_node.size / acc_rej_test_max_fan_out))
                if not should_continue:
                    return None, walking_path, None


            walking_path.append((next_random_step, current_node.size, acc_rej_test_max_fan_out))

        next_random_step = np.random.randint(low=0, high=current_node.size)
        walking_path.append((next_random_step, current_node.size, acc_rej_test_max_fan_out))

        leaf = current_node._keys
        return leaf[next_random_step], walking_path, acc_rej_test_acceptance_prob

    def _get_path_to_bucket_of_value(self, key):
        path = []
        current_node = self._data

        index = self._search(key)
        # TODO: im not sure what _search('M') return 0, check it
        if index >= 0:  # TODO: else?
            if isinstance(self._data[index].child, self._bucket_type):
                index_in_leaf = _find_first_greather_value_in_list(self._data[index].child._values,
                    key)
                return [index, index_in_leaf]
            path_from_child = self._data[index].child._get_path_to_bucket_of_value(key)
            path_from_child = [index] + path_from_child
        else:
            raise  # TODO!
        return path_from_child

    def _random_next_move_respect_fanout_prob(self, current_node, walking_path):
        walking_path_str = str(walking_path)
        if walking_path_str in self.walking_path_to_fanout_distribution:
            node_distribution = self.walking_path_to_fanout_distribution[walking_path_str]
        else:
            all_sizes = np.array([node.child.size for node in current_node._data])
            node_distribution = all_sizes / sum(all_sizes)
            self.walking_path_to_fanout_distribution[walking_path_str] = node_distribution

        return np.random.choice(current_node.size, p=node_distribution)


    def join(self, right_tree):
        pass


def _find_first_greather_value_in_list(sorted_list, key):
    values_greater_than_key = [index for index, val in enumerate(sorted_list) if val >= key]
    if values_greater_than_key:
        return values_greater_than_key[0]
    return len(sorted_list) - 1


def add_to_debug_global(all_vars):
    global _debug_random_sampling
    _debug_random_sampling.append({
        'params': {
            'k': all_vars['k'],
            'how_to_walk': all_vars['how_to_walk'],
        },
        'all_accept_reject_measures': all_vars['all_accept_reject_measures']
    })


def _this_value_was_sampled_already(walking_path, all_walking_paths_set):
    return str(walking_path) in all_walking_paths_set


def _get_max_fan_out_for_(current_node):
    return np.max([node_data.child.size for node_data in current_node._data])


def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob

