from BTrees.OOBTree import OOBTree as _OOBTree
import numpy as np

from btrees.btree_sampling_olken import OOBTreeExtOlken
from btrees.btree_sampling_ours import OOBTreeExtOur


class OOBTreeExt(OOBTreeExtOlken, OOBTreeExtOur):

    def _persist_sampling_stats(self):
        pass

def _accept_reject_test_pass(acceptance_prob):
    rand_num = np.random.random_sample()
    return rand_num < acceptance_prob


