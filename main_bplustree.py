import bplustree
import uuid
from bplustree import BPlusTree, UUIDSerializer


def main():
    tree = bplustree.BPlusTree(filename='/tmp/btree_idx.idx')
    tree = BPlusTree('/tmp/bplustree.db', serializer=UUIDSerializer(), key_size=16)
    tree.insert(uuid.uuid1(), b'foo')

    pass
if __name__ == '__main__':
    main()
