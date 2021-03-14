import os
os.environ["PURE_PYTHON"] = "True"
from BTrees.OOBTree import OOBTree as _OOBTree
import numpy as np

DEFAULT_EXPLORING_STEP = 0

class OOBTreeExtOur(_OOBTree):
    def __init__(self):
        super(OOBTreeExtOur, self).__init__()
        self._fanout_distribution_cache = {}

    def sample_distribution_height_three(self, k):
        self._fanout_distribution_cache = {}
        sampled_values = []
        sampled_paths = []

        while len(sampled_values) < k:
            value, path = self._our_walk_random_root_to_item(node=self)
            if path in sampled_paths:
                continue # todo: revist counter

            sampled_paths.append(path)
            sampled_values.append(value)

        return sampled_values

    def _our_walk_random_root_to_item(self, node):
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

    def sample_distribution_height_4(self, k):
        pass

    def _calc_fanout_distribution_of_node(self, node):
        if node in self._fanout_distribution_cache:
            self._cache_hit_counter['hit'] += 1
            return self._fanout_distribution_cache[node].copy()
        self._cache_hit_counter['miss'] += 1

        all_sizes = np.array([node.child.size for node in node._data])
        node_distribution = all_sizes / sum(all_sizes)

        self._fanout_distribution_cache[node] = node_distribution
        return node_distribution.copy()