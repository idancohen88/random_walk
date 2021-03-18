import numpy as np

from btrees.btree_sampling_olken import OOBTreeExtOlken
from btrees.btree_sampling_distribution_oriented import OOBTreeExtFanoutOriented


class OOBTreeExt(OOBTreeExtOlken, OOBTreeExtFanoutOriented):

    def _persist_sampling_stats(self):
        pass

    def _get_random_step_and_value_from_bucket(self, bucket):
        next_random_step = np.random.randint(low=0, high=bucket.size)

        value_in_leaf = bucket.items()[next_random_step]
        return next_random_step, value_in_leaf

def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob

