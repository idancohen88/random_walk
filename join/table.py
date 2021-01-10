import uuid
from collections import Counter
import random
import numpy as np

def _full_join(table_left, table_right):
    return [(i[0], i[1], j[1]) for i in table_left for j in table_right if i[0] == j[0]]


class Table:
    _freq = None
    _keys = None
    _values = None

    def __init__(self, name, *args):
        self.name = name
        if isinstance(args[0], list):
            our_data = args[0]
        else:
            our_data = args

        self._values = [(x, uuid.uuid4().hex) for x in our_data]

    def __iter__(self):
        yield from self._values

    @property
    def freq(self):
        if not self._freq:
            self._freq = Counter(self.keys)
        return self._freq

    @property
    def keys(self):
        if not self._keys:
            self._keys = [x[0] for x in self._values]
        return self._keys

    def rand(self):
        return random.sample(self._values, 1)[0]

    def weighted_rand(self, other_tbl_freq):
        weights = np.array([other_tbl_freq[x[0]] for x in self._values])
        prob_weights = weights / sum(weights)
        rand_index = np.random.choice(len(self._values), p=prob_weights)
        return self._values[rand_index]

    def join(self, tbl):
        return _full_join(self, tbl)

    def get_key(self, value):
        return [x for x in self._values if x[0] == value]