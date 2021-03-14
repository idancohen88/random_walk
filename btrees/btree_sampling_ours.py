from BTrees.OOBTree import OOBTree as _OOBTree
import numpy as np


class OOBTreeExtOur(_OOBTree):
    def __init__(self):
        super(OOBTreeExtOur, self).__init__()
        self.walking_path_to_fanout_distribution = {}

    def sample_distribution_height_three(self, k):
        self.walking_path_to_fanout_distribution = {}
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
            next_random_step = self._random_next_move_respect_fanout_prob(
                current_node, walking_path
            )
            current_node = current_node._data[next_random_step].child

            walking_path.append(next_random_step)

        next_random_step, value_in_leaf = self._get_random_step_and_value_from_bucket(bucket=current_node)
        walking_path.append(next_random_step)
        return value_in_leaf, walking_path

    def _random_next_move_respect_fanout_prob(self, current_node, walking_path):
        walking_path_str = str(walking_path)
        if walking_path_str in self.walking_path_to_fanout_distribution:
            node_distribution = self.walking_path_to_fanout_distribution[walking_path_str]
        else:
            all_sizes = np.array([node.child.size for node in current_node._data])
            node_distribution = all_sizes / sum(all_sizes)
            self.walking_path_to_fanout_distribution[walking_path_str] = node_distribution

        return np.random.choice(current_node.size, p=node_distribution)

    def sample_distribution_height_4(self, k):
        pass
