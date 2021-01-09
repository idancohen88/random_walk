import random
import string
import os
#

from collections import Counter
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
KEY_LENGTH = 8
NUM_OF_ROWS = 8000000

def main2():
    from bplustree import BPlusTree
    my_tree = BPlusTree(filename=os.getcwd() + '/my_db.file')
    random_values = [("".join(random.choices(string.ascii_lowercase , k=8)), "value") for _ in range(10)]
    random_values = sorted(random_values)
    my_tree.batch_insert(random_values)

    print('')

def main():
    from btree_pingf import BPlusTree
    random_values =[(''.join(random.choice(string.ascii_lowercase) for _ in range(KEY_LENGTH)), 'value') for x in range(NUM_OF_ROWS)]
    random_values = sorted(random_values, key=lambda x:x[0])
    random_values = [(x,x) for x in random_values]
    my_tress = BPlusTree.bulkload(random_values, order=250)
    # my_tress._root.children[0].children[0].children[0]
    # dict(Counter([len(level_2.children) for level_1 in my_tress._root.children for level_2 in level_1.children]))
    print('')

if __name__ == '__main__':
    main()