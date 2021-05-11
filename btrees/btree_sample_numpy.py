from datetime import datetime
import numpy as np
from btrees.btree_base import OOBTreeBase


class OOBTreeExtNumpy(OOBTreeBase):
    def sample_numpy(self, k):
        start_time = datetime.now()
        k = self._min_between_k_and_btree_size(k)

        sampled_indx = np.random.choice(len(self), k)
        all_values = np.array(list(self.iteritems()))
        sampled_values = all_values[sampled_indx]
        self._persist_sampling_stats(
            start_time=start_time,
            sampled_values=sampled_values,
            name="numpy",
            sample_size=k,
        )

        return sampled_values
