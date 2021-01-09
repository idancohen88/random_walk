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



def _sample(tbl1, tbl2):
    tbl1_participant = sum([tbl2.freq[x] for x in set(tbl1.keys)])
    tbl2_participant = sum([tbl1.freq[x] for x in set(tbl2.keys)])
    chosen_elements = defaultdict(lambda: {'join_freq': 0, 'sampled': 0, 'sampled_from': [], 'iteration': 0})
    freq_in_join_estimate = {key: value * tbl2.freq[key] for key, value in tbl1.freq.items() if value * tbl2.freq[key]}
    #freq_in_join_estimate_per_tbl = {key: value * tbl2.freq[key] for key, value in tbl1.freq.items() if value * tbl2.freq[key]}

    sampled = 0
    i = 0
    K = 100

    samples = []
    full_join = tbl1.join(tbl2)
    print('we have full join')
    values_in_join = {x[0] for x in full_join}
    tables_picker = _tables_iterator(tbl1, tbl2)
    sampled_elements = []
    declined = []
    while sum([x['sampled'] for x in chosen_elements.values()]) < K:
        i += 1
        curr_table, other_table = next(tables_picker)

        print('curr table selected')
        element = curr_table.rand()
        element_key = element[0]

        while not _sampled_element_is_valid(element, curr_table, sampled_elements, other_table):
            declined.append(element)
            element = curr_table.rand()
            element_key = element[0]

        print('we have curr table')
        chosen_elements[element_key]['join_freq'] = freq_in_join_estimate[element_key]


        chosen_elements[element_key]['sampled'] += 1
        chosen_elements[element_key]['sampled_from'].append(curr_table.name)
        sampled_elements.append((element, curr_table.name))
        print(f'sampled element {element} from table {curr_table.name}')

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


