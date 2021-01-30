import os

os.environ["PURE_PYTHON"] = "True"
from BTrees import check
from BTrees.OOBTree import OOBTree

from btree_ext_lean import OOBTreeExtLean
from build_tree import generate_btree_index_x_values_with_dist

prefix_to_percent = {
    'gggg': 0.25,
    'hhhh': 0.15,
    'mmmm': 0.10,
    'rrrr': 0.03,
    '': 0.47
}
num_of_values = 4_000_000
#num_of_values = 200_000
my_index = generate_btree_index_x_values_with_dist(num_of_values, prefix_to_percent, OOBTreeExtLean())

#my_index_2 = OOBTree()
#my_index_2.update(my_index.items())

while (True):
    my_index.random_sampling(k=10_000)

#my_index.get('AEleUQiL')
print(1)


