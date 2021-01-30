import os

DEFAULT_EXPLORING_STEP = 0
os.environ["PURE_PYTHON"] = "True"

from BTrees.OOBTree import OOBTree as _OOBTree
from collections import Counter
import numpy as np


_debug_random_sampling = []


class OOBTreeExtLean(_OOBTree):
    _fanout_distribution_cache = {}
    _cache_hit_counter = Counter()

    def __init__(self):
        super(OOBTreeExtLean, self).__init__()
        self.walking_path_to_fanout_distribution = {}
        self.default_exploring_step = DEFAULT_EXPLORING_STEP

    def random_sampling(self, k):
        self._first_walk_to_determine_root_coefs()
        self.walking_path_to_fanout_distribution = {}
        all_accept_reject_measures = {
            'accept': [],
            'reject': [],
            'revisited_paths': Counter()
        }

        k = min(len(self), k)
        sampled_values = []
        all_walking_paths_set = set()
        all_walking_paths_stats = []
        while len(sampled_values) < k:
            sampled_value, walking_path, walking_path_stats  = \
                self._get_value_and_path_by_random_walk_from_node(node=self)

            if _this_value_was_sampled_already(walking_path, all_walking_paths_set):
                all_accept_reject_measures['revisited_paths'][str(walking_path)] += 1
                continue

            accept_reject_measures = {
                'path': walking_path,
                'value': sampled_value,
            }

            all_accept_reject_measures['accept'].append(accept_reject_measures)

            all_walking_paths_set.add(str(walking_path))
            all_walking_paths_stats.append(walking_path_stats)
            sampled_values.append(sampled_value)

        add_to_debug_global(locals())

        return sampled_values


    def _calc_fanout_distribution_of_node(self, node):
        if node in self._fanout_distribution_cache:
            self._cache_hit_counter['hit'] += 1
            return self._fanout_distribution_cache[node]
        self._cache_hit_counter['miss'] += 1

        all_sizes = np.array([node.child.size for node in node._data])
        node_distribution = all_sizes / sum(all_sizes)

        self._fanout_distribution_cache[node] = node_distribution
        return node_distribution

    def _first_walk_to_determine_root_coefs(self):
        branch_coefs = self._determine_root_to_leaf_walking_probs(root=self)
        equations_matrix, equations_equal_matrix = self._create_equations_for_equaling_all_walking_probs(branch_coefs)

        self.root_probs_coefs = np.linalg.solve(equations_matrix, equations_equal_matrix)

    def _create_equations_for_equaling_all_walking_probs(self, branch_coefs):
        root_fanout_distribution = self._calc_fanout_distribution_of_node(node=self)

        equations_matrix = np.zeros((len(branch_coefs) + 1, len(branch_coefs)))
        equations_equal_matrix = np.zeros(len(branch_coefs))

        for root_child_number in range(len(branch_coefs)):
            equations_matrix[root_child_number][0] = branch_coefs[0]
            equations_matrix[root_child_number][root_child_number] = -1 * branch_coefs[root_child_number]
            equations_matrix[-1][root_child_number] = root_fanout_distribution[root_child_number]

        equations_matrix = equations_matrix[1:, ]

        equations_equal_matrix[-1] = 1
        return equations_matrix, equations_equal_matrix

    def _determine_root_to_leaf_walking_probs(self, root):
        root_to_leaf_walking_probs = {}
        root_fanout_distribution = self._calc_fanout_distribution_of_node(root)

        for root_child_number in range(len(root._data)):
            current_node = root
            walking_prob = root_fanout_distribution[root_child_number]
            current_node = current_node._data[root_child_number].child

            while not isinstance(current_node._data[DEFAULT_EXPLORING_STEP].child, self._bucket_type):
                node_fanout_distribution = self._calc_fanout_distribution_of_node(current_node)
                walking_prob *= node_fanout_distribution[0]
                current_node = current_node._data[DEFAULT_EXPLORING_STEP].child

            assert isinstance(current_node._data[DEFAULT_EXPLORING_STEP].child, self._bucket_type)
            walking_prob *= 1 / len(current_node._data)
            root_to_leaf_walking_probs[root_child_number] = walking_prob

        branch_coefs = np.array(list(root_to_leaf_walking_probs.values()))
        return branch_coefs

    def _get_value_and_path_by_random_walk_from_node(self, node):
        walking_path = []
        current_node = node
        prob_along_path = 1
        walking_path_stats = []
        while not isinstance(current_node, self._bucket_type):
            root = not walking_path_stats
            if root:
                next_random_step, chosen_random_step_prob = self._random_next_move_respect_fanout_prob_from_root(
                    current_node, walking_path)
            else:
                next_random_step, chosen_random_step_prob = self._random_next_move_respect_fanout_prob(current_node, walking_path)
            prob_along_path *= chosen_random_step_prob
            walking_path.append((next_random_step, current_node.size, chosen_random_step_prob, prob_along_path))
            current_node = current_node._data[next_random_step].child
            walking_path_stats.append({
                'next_random_step': next_random_step,
                'chosen_random_step_prob':
                    chosen_random_step_prob, 'prob_along_path':prob_along_path})

        next_random_step = np.random.randint(low=0, high=current_node.size)
        chosen_random_step_prob = 1/current_node.max_leaf_size  # todo: size
        prob_along_path *= chosen_random_step_prob
        walking_path.append((next_random_step, current_node.size, chosen_random_step_prob, prob_along_path))
        walking_path_stats.append({
            'next_random_step': next_random_step,
            'chosen_random_step_prob':
                chosen_random_step_prob, 'prob_along_path': prob_along_path,
            'entire_walking_path': walking_path})

        leaf = current_node._keys
        return leaf[next_random_step], walking_path, walking_path_stats


    def _random_next_move_respect_fanout_prob(self, current_node, walking_path):
        walking_path_str = str(walking_path)
        if walking_path_str in self.walking_path_to_fanout_distribution:
            node_distribution = self.walking_path_to_fanout_distribution[walking_path_str]
        else:
            all_sizes = np.array([node.child.size for node in current_node._data])
            node_distribution = all_sizes / sum(all_sizes)
            self.walking_path_to_fanout_distribution[walking_path_str] = node_distribution

        next_random_step = np.random.choice(current_node.size, p=node_distribution)
        chosen_random_step_prob = node_distribution[next_random_step]
        return next_random_step, chosen_random_step_prob

    def _random_next_move_respect_fanout_prob_from_root(self, current_node, walking_path):
        walking_path_str = str(walking_path)
        if walking_path_str in self.walking_path_to_fanout_distribution:
            node_distribution = self.walking_path_to_fanout_distribution[walking_path_str]
        else:
            all_sizes = np.array([node.child.size for node in current_node._data])
            node_distribution = np.array(all_sizes / sum(all_sizes))
            node_distribution *= self.root_probs_coefs
            #node_distribution = node_distribution / sum(node_distribution)
            self.walking_path_to_fanout_distribution[walking_path_str] = node_distribution
        next_random_step = np.random.choice(current_node.size, p=node_distribution)
        chosen_random_step_prob = node_distribution[next_random_step]
        return next_random_step, chosen_random_step_prob


    def join(self, right_tree):
        pass


def add_to_debug_global(all_vars):
    global _debug_random_sampling
    _debug_random_sampling.append({
        'params': {
            'k': all_vars['k'],
        },
        'tree_size': len(all_vars['self']),
        'all_accept_reject_measures': all_vars['all_accept_reject_measures'],
        'all_walking_paths_stats': all_vars['all_walking_paths_stats']
    })


def _this_value_was_sampled_already(walking_path, all_walking_paths_set):
    return str(walking_path) in all_walking_paths_set

