import numpy as np

from join.table import Table
import random
from collections import defaultdict, Counter


def _tables_iterator(table1, table2):
    tbl1_participant = sum([table2.freq[x] for x in set(table1.keys)])
    tbl2_participant = sum([table1.freq[x] for x in set(table2.keys)])
    sum_of_join = tbl1_participant + tbl2_participant
    while True:
        chosen = np.random.choice([1,2], p=[tbl1_participant/sum_of_join, tbl2_participant/sum_of_join])
        if chosen == 1:
            yield table1, table2
        else:
            yield table2, table1


def _sampled_element_is_valid(element, curr_table, sampled_elements, other_table):
    element_key = element[0]
    specific_element_sampled = sum([sampled_element == element and table_name==curr_table.name for sampled_element, table_name in sampled_elements])

    max_sampling_allowed_for_element_from_curr_table = other_table.freq[element_key]
    return specific_element_sampled < max_sampling_allowed_for_element_from_curr_table



def _sample(tbl1, tbl2, k=50):
    chosen_elements = defaultdict(lambda: {'join_freq': 0, 'sampled': 0, 'sampled_from': [], 'iteration': 0})
    freq_in_join_estimate = {key: value * tbl2.freq[key] for key, value in tbl1.freq.items() if value * tbl2.freq[key]}
    sum_of_join = sum(freq_in_join_estimate.values())

    rejects = []

    while sum([x['sampled'] for x in chosen_elements.values()]) < k:
        element = tbl1.rand()
        element_key = element[0]

        accept_prob = freq_in_join_estimate.get(element_key, 0 )/ sum_of_join
        accept_test_pass = np.random.random_sample() <= accept_prob

        if not accept_test_pass:
            rejects.append(element_key)
            continue

        chosen_elements[element_key]['join_freq'] = freq_in_join_estimate[element_key]


        chosen_elements[element_key]['sampled'] += 1


    print('done sampling')

def _calculate_distribution(values):
    return {value: occurences/len(values) for value, occurences in Counter([key for key in values]).most_common(10)}


def main():
    tbl1 = Table('tbl1',[random.randint(1, 10) for _ in range(50)] +
                 [random.randint(10, 20) for _ in range(20)] + [13 for _ in
                                                                range(60)])
    tbl2 = Table('tbl2', [random.randint(10, 20) for _ in range(53)])

    _sample(tbl1, tbl2)

if __name__ == '__main__':
    main()


