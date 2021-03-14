import os


os.environ["PURE_PYTHON"] = "True"

from btrees.btree_ext_deprecated import OOBTreeExt, WALKING_TECHNIQUE
from build_tree.build_tree import generate_btree_index_x_values_with_dist

prefix_to_percent = {
    'gggg': 0.25,
    'hhhh': 0.15,
    'mmmm': 0.10,
    'rrrr': 0.03,
    '': 0.47
}
#num_of_values = 4_000_000
num_of_values = 200_000
my_index = generate_btree_index_x_values_with_dist(num_of_values, prefix_to_percent, OOBTreeExt())

#my_index_2 = OOBTree()
#my_index_2.update(my_index.items())

while (True):
    sampled = my_index.random_sampling(k=10_000, how_to_walk=WALKING_TECHNIQUE.RANDOM_WITH_TEST_EARLY_ABORT)
    pass

#my_index.get('AEleUQiL')
print(1)


