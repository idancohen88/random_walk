import os

from build_tree.build_tree import ALPHABET

os.environ["PURE_PYTHON"] = "True"

from btrees.btee_ext_new import OOBTreeExt

MAX_INTERNAL_SIZE = 4
MAX_LEAF_SIZE = 4

def test_oklen_sanity():
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_olken(3)) == 3

def test_oklen__early_abort_sanity():
    my_index = _generate_3_height_btree()
    assert len(my_index.sample_olken_early_abort(3)) == 3


def _generate_3_height_btree():
    my_index = OOBTreeExt()
    my_index.max_internal_size = MAX_INTERNAL_SIZE
    my_index.max_leaf_size = MAX_LEAF_SIZE
    my_index.has_key('a')
    for i, c in enumerate(ALPHABET):
        my_index[c] = i

    assert isinstance(my_index._data[0].child._data[0].child, my_index._bucket_type), (
        '3 height btree must arrive to bucket after wo steps')
    return my_index



if __name__ == '__main__':
    test_oklen_sanity()
    test_oklen__early_abort_sanity()