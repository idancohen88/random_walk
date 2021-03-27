from btrees.btree_sampling_btwrs import OOBTreeExtBtwrs
from btrees.btree_sampling_olken import OOBTreeExtOlken
from btrees.btree_sampling_distribution_oriented import OOBTreeExtFanoutOriented


class OOBTreeExt(OOBTreeExtOlken, OOBTreeExtFanoutOriented, OOBTreeExtBtwrs):
    pass

