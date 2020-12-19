from btree_ext import WALKING_TECHNIQUE
from build_tree import generate_btree_index_x_values_with_dist

prefix_to_percent = {
    'gggg': 0.25,
    'hhhh': 0.15,
    'mmmm': 0.10,
    'rrrr': 0.03,
    '': 0.47
}
num_of_values = 500_000
my_index = generate_btree_index_x_values_with_dist(num_of_values, prefix_to_percent)
while (True):
    my_index.random_sampling(k=100, how_to_walk=WALKING_TECHNIQUE.DISTRIBUTION_ORIENTED)

#my_index.get('AEleUQiL')
print(1)


