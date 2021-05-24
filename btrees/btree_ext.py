from btrees.btree_sample_monkey import OOBTreeExtMonkey
from btrees.btree_sample_numpy import OOBTreeExtNumpy
from btrees.btree_sampling_btwrs import OOBTreeExtBtwrs
from btrees.btree_sampling_olken import OOBTreeExtOlken
from btrees.btree_sampling_distribution_oriented import OOBTreeExtFanoutOriented
from btrees.naive_random_walk import OOBTreeExtNaiveRandomWalk


class OOBTreeExt(OOBTreeExtOlken, OOBTreeExtFanoutOriented, OOBTreeExtBtwrs, OOBTreeExtMonkey, OOBTreeExtNumpy,
                 OOBTreeExtNaiveRandomWalk):
    pass

