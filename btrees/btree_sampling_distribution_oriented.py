from collections import Counter
from datetime import datetime
import numpy as np

from btrees.btee_base import OOBTreeBase

DEFAULT_EXPLORING_STEP = 0

class OOBTreeExtFanoutOriented(OOBTreeBase):
    def __init__(self):
        super(OOBTreeExtFanoutOriented, self).__init__()
        self._fanout_distribution_cache = {}
        self._cache_hit_counter = Counter()

    def sample_distribution_oriented_height_three(self, k):
        start_time = datetime.now()
        self._fanout_distribution_cache = {}
        self._cache_hit_counter = Counter()
        sampled_values = []
        sampled_paths = []

        while len(sampled_values) < k:
            value, path = self._fanout_oriented_random_walk_node_to_item(node=self)
            if path in sampled_paths:
                continue # todo: revist counter

            sampled_paths.append(path)
            sampled_values.append(value)

        self._persist_sampling_stats(start_time=start_time,sampled_values=sampled_values,
                                     name='distribution_oriented_height_three', sample_size=k)
        return sampled_values

    def _fanout_oriented_random_walk_node_to_item(self, node):
        current_node = node
        walking_path = []

        while not isinstance(current_node, self._bucket_type):
            next_random_step = self._random_next_move_respect_fanout_prob(current_node)
            current_node = current_node._data[next_random_step].child
            walking_path.append(next_random_step)

        next_random_step, value_in_leaf = self._get_random_step_and_value_from_bucket(bucket=current_node)
        walking_path.append(next_random_step)
        return value_in_leaf, walking_path

    def _random_next_move_respect_fanout_prob(self, current_node):
        node_distribution = self._calc_fanout_distribution_of_node(current_node)

        return np.random.choice(current_node.size, p=node_distribution)

    def sample_distribution_oriented_height_four(self, k):
        start_time = datetime.now()
        self._fanout_distribution_cache = {}
        self._cache_hit_counter = Counter()
        root = self
        root_probs_coefs = self._first_walk_to_determine_root_coefs() # todo:
        sampled_values = []
        sampled_paths = []

        while len(sampled_values) < k:
            next_random_step = np.random.choice(root.size, p=root_probs_coefs)
            path = [next_random_step]
            current_node = root._data[next_random_step].child
            value, walking_path_3_height = self._fanout_oriented_random_walk_node_to_item(current_node)
            path.extend(walking_path_3_height)

            if path in sampled_paths:
                continue # todo: revist counter

            sampled_values.append(value)
            sampled_paths.append(path)

        self._persist_sampling_stats(start_time=start_time, sampled_values=sampled_values,
                                     name='distribution_oriented_height_four', sample_size=k)
        return sampled_values

    def _first_walk_to_determine_root_coefs(self):
        branch_coefs = self._determine_root_to_leaf_walking_probs(root=self)
        equations_matrix, equations_equal_matrix = self._create_equations_for_equaling_all_walking_probs(branch_coefs)

        return np.linalg.solve(equations_matrix, equations_equal_matrix)

    def _extract_node_to_leaf_probability(self, node):
        walking_prob = 1
        current_node = node
        while not isinstance(current_node, self._bucket_type):
            node_fanout_distribution = self._calc_fanout_distribution_of_node(current_node)
            walking_prob *= node_fanout_distribution[DEFAULT_EXPLORING_STEP]
            current_node = current_node._data[DEFAULT_EXPLORING_STEP].child

        assert isinstance(current_node, self._bucket_type)
        walking_prob *= 1 / current_node.size
        return walking_prob


    def _determine_root_to_leaf_walking_probs(self, root):
        return [self._extract_node_to_leaf_probability(node=child_of_root.child)
                for child_of_root in root._data]

    def _create_equations_for_equaling_all_walking_probs(self, branch_coefs):
        equations_matrix = np.zeros((len(branch_coefs) + 1, len(branch_coefs)))
        equations_equal_matrix = np.zeros(len(branch_coefs))

        for root_child_number in range(len(branch_coefs)):
            equations_matrix[root_child_number][0] = branch_coefs[0]
            equations_matrix[root_child_number][root_child_number] = -1 * branch_coefs[root_child_number]
            equations_matrix[-1][root_child_number] = 1

        equations_matrix = equations_matrix[1:, ]

        equations_equal_matrix[-1] = 1
        return equations_matrix, equations_equal_matrix

    def _calc_fanout_distribution_of_node(self, node):
        if node in self._fanout_distribution_cache:
            self._cache_hit_counter['hit'] += 1
            return self._fanout_distribution_cache[node] # todo: why I had .copy()?
        self._cache_hit_counter['miss'] += 1

        all_sizes = np.array([node.child.size for node in node._data])
        node_distribution = all_sizes / sum(all_sizes)

        self._fanout_distribution_cache[node] = node_distribution
        return node_distribution.copy()