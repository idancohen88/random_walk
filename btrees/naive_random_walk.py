from datetime import datetime

import numpy as np
from btrees.btree_base import OOBTreeBase


class OOBTreeExtNaiveRandomWalk(OOBTreeBase):
    def sample_naive_random_walk(self, k):
        start_time = datetime.now()
        k = self._min_between_k_and_btree_size(k)

        sampled_values = []
        sampled_paths = []
        while len(sampled_values) < k:
            value, path = self._naive_random_walk_to_leaf()

            if path in sampled_paths:
                continue

            sampled_values.append(value)
            sampled_paths.append(path)

        self._persist_sampling_stats(
            start_time=start_time,
            sampled_values=sampled_values,
            name="naive_random_walk",
            sample_size=k,
        )

        self.save_sampled_path(name="naive_random_walk",
                               k=k,
                               sampled_paths=sampled_paths)

        return sampled_values

    def _naive_random_walk_to_leaf(self):
        walking_path = []
        current_node = self

        while not isinstance(current_node, self._bucket_type):
            next_random_step = np.random.randint(low=0, high=current_node.size)
            current_node = current_node._data[next_random_step].child

            walking_path.append(next_random_step)

        next_random_step = np.random.randint(low=0, high=current_node.size)
        value_in_leaf = current_node.items()[next_random_step]

        walking_path.append(next_random_step)
        return value_in_leaf, walking_path

    def _get_random_step_and_value_from_bucket(self, bucket):
        next_random_step = np.random.randint(low=0, high=bucket.size)

        value_in_leaf = bucket.items()[next_random_step]
        return next_random_step, value_in_leaf

