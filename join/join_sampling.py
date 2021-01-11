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


def _sampling_like_reservoirs(candidates_appearances, s1_freq):
    all_sampled_elements = []
    for key, all_elements in candidates_appearances.items():
        sample_size = s1_freq[key]
        sampled_elements_indexes = np.random.choice(len(all_elements), size=sample_size, replace=True)
        sampled_elements = [all_elements[index] for index in sampled_elements_indexes]
        all_sampled_elements.extend(sampled_elements)

    return all_sampled_elements


def _sample2(tbl1, tbl2, k=50):
    sampled = []

    sampled_tbl1 = _extract_k_candidate_tbl1(k, tbl1, tbl2)
    sampled_tbl1_counter = Counter([x[0] for x in sampled_tbl1])

    candidates_appearances_from_tbl2 = _extract_all_candidates_from_tbl2(sampled_tbl1_counter, tbl2)
    tbl2_sampled = _sampling_like_reservoirs(candidates_appearances_from_tbl2, sampled_tbl1_counter)

    sampled = _match_tupples_for_final_sampling(sampled_tbl1, tbl2_sampled)

    print()


def _match_tupples_for_final_sampling(sampled_tbl1, tbl2_sampled):
    sampled = []
    for element_tbl1 in sampled_tbl1:
        element_tbl2 = next(filter(lambda x: x[0] == element_tbl1[0], tbl2_sampled))
        tbl2_sampled.remove(element_tbl2)
        sampled.append((element_tbl1, element_tbl2))
    return sampled

def _reservoirs_sample(element_key, r, n):
    pass


def _extract_all_candidates_from_tbl2(sampled_tbl1_counter, tbl2):
    many_reservoirs = defaultdict(list)
    for element in tbl2:
        element_key = element[0]
        if not element_key in sampled_tbl1_counter:
            continue

        #_reservoirs_sample(element_key, r=sampled_tbl1_counter[element_key], n=tbl2.freq[element_key])
        many_reservoirs[element_key].append(element)

    return many_reservoirs

def _extract_k_candidate_tbl1(k, tbl1, tbl2):
    return [tbl1.weighted_rand(tbl2.freq) for _ in range(k)]


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


    _sample2(tbl1, tbl2)

if __name__ == '__main__':
    main()


