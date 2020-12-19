# from btree_ext import OOBTreeExt
# from datasketch import MinHash
import numpy as np
import random
from collections import Counter, defaultdict


def sample_join_method_a(table1, table2, sample_size=50):
    table1 = np.array(table1) if not isinstance(table1, np.ndarray) else table1
    table2 = np.array(table2) if not isinstance(table2, np.ndarray) else table2
    random_join = []
    max_int = len(table1) - 1
    already_visited_table1 = set()

    while len(random_join) < sample_size and len(already_visited_table1) < len(table1):
        next_rand_index = random.randint(0, max_int)
        while next_rand_index in already_visited_table1:
            next_rand_index = random.randint(0, max_int)

        already_visited_table1.add(next_rand_index)
        value_from_rand_index = table1[next_rand_index]

        joined_indexes_from_table2 = np.where(table2 == value_from_rand_index)
        if not joined_indexes_from_table2:
            continue
        rand_matched_index_table2 = random.choice(joined_indexes_from_table2)[0]
        # if rand_matched_index_table2.size:
        rand_value_table2 = table2[rand_matched_index_table2]
        random_join.append((value_from_rand_index, rand_value_table2))

    print(Counter([x[0] for x in random_join]))


def sample_join_method_b(table1, table2, sample_size=50):
    table1 = np.array(table1) if not isinstance(table1, np.ndarray) else table1
    table2 = np.array(table2) if not isinstance(table2, np.ndarray) else table2
    random_join = []
    max_int = len(table1) - 1
    already_visited_table1 = set()

    table_2_counter = Counter(table2)
    max_cardinality = table_2_counter.most_common(1)[0][1]

    while len(random_join) < sample_size and len(already_visited_table1) < len(table1):
        next_rand_index = random.randint(0, max_int)
        while next_rand_index in already_visited_table1:
            next_rand_index = random.randint(0, max_int)

        value_from_rand_index = table1[next_rand_index]

        joined_indexes_from_table2 = np.where(table2 == value_from_rand_index)
        if not joined_indexes_from_table2:
            continue

        olken_accept_test = table_2_counter[value_from_rand_index] / max_cardinality

        if np.random.random_sample() >= olken_accept_test:
            continue

        # already_visited_table1.add(next_rand_index)

        rand_matched_index_table2 = random.choice(joined_indexes_from_table2)[0]

        rand_value_table2 = table2[rand_matched_index_table2]
        random_join.append((value_from_rand_index, rand_value_table2))

    print(Counter([x[0] for x in random_join]))


def sample_candidate_from_table(table, do_not_touch):
    # todo: revisited
    max_int = len(table) - 1
    rand_index = random.randint(0, max_int)
    while rand_index in do_not_touch and len(table) > len(do_not_touch):
        rand_index = random.randint(0, max_int)
    return rand_index, table[rand_index]


def randomly_pick_tuple_with_value_from_table(table, value, index_to_using_stats):
    matches_indexes_of_value = np.where(table == value)
    matches_indexes_of_value = matches_indexes_of_value[0]
    # if not matches_indexes_of_value:
    #     return
    #
    choose_good_one = False

    indexes_cant_touch = {
        index
        for index, using_stats in index_to_using_stats.items()
        if using_stats["allow"] == using_stats["used"]
    }

    while matches_indexes_of_value.size and not choose_good_one:
        rand_index_of_match_values = random.choice(matches_indexes_of_value)
        if rand_index_of_match_values not in indexes_cant_touch:
            choose_good_one = True
        else:
            matches_indexes_of_value = np.delete(matches_indexes_of_value, rand_index_of_match_values)

    if not choose_good_one:
        # todo: also update do not touch, for performance
        return None

    index_to_using_stats[rand_index_of_match_values]["used"] += 1
    return rand_index_of_match_values, table[rand_index_of_match_values]


def _build_index_to_using_stats(indexes_table, freq_table, sample_ratio):
    # no need doing that in advanced, but just for saving complexity.
    usage_stags = Counter(freq_table)

    return {
        i: {"used": 0, "allow": round(usage_stags[value] * sample_ratio) or 1, "original_value": value}
        for i, value in enumerate(indexes_table)
    }


def sample_join_method_c(table_left, table_right, sample_size=50):
    table_left = np.array(table_left) if not isinstance(table_left, np.ndarray) else table_left
    table_right = np.array(table_right) if not isinstance(table_right, np.ndarray) else table_right
    random_join = []

    sample_ratio = sample_size / len([(i, j) for i in table_left for j in table_right if i == j])

    index_to_using_stats_in_table_left = _build_index_to_using_stats(
        indexes_table=table_left, freq_table=table_right, sample_ratio=sample_ratio
    )
    index_to_using_stats_in_table_right = _build_index_to_using_stats(
        indexes_table=table_right, freq_table=table_left, sample_ratio=sample_ratio
    )

    table_left_do_not_touch = set()
    table_right_do_not_touch = set()

    sample_round_to_side_mapping = {
        0: {
            'table1': table_left,
            'table2': table_right,
            'do_not_touch': table_left_do_not_touch,
            'index_to_using_stats': index_to_using_stats_in_table_right,
            'index_to_using_stats2': index_to_using_stats_in_table_left,
            '_name': 'sample_left_and_join_right'
        },
        1: {
            'table1': table_right,
            'table2': table_left,
            'do_not_touch': table_right_do_not_touch,
            'index_to_using_stats': index_to_using_stats_in_table_left,
            'index_to_using_stats2': index_to_using_stats_in_table_right,
            '_name': 'sample_right_and_join_left'
        }
    }

    i=0

    while len(random_join) < sample_size:
        # current_round = len(random_join) % 2
        current_round = 0
        i+=1
        #c current_round_mapping = sample_round_to_side_mapping[current_round]

        sampled_tuple = sample_from_table_1_and_join_table_2(
            table1=table_left,
            table2=table_right,
            index_to_using_stats_table_2=index_to_using_stats_in_table_right,
            index_to_using_stats_table_1=index_to_using_stats_in_table_left)

        if sampled_tuple:
            random_join.append(sampled_tuple)


    print(Counter([x[0] for x in random_join]))


def sample_from_table_1_and_join_table_2(table1, table2, do_not_touch_table1, index_to_using_stats_table_2):
    candidate_index, candidate_value = sample_candidate_from_table(table=table1, do_not_touch=do_not_touch_table1)
    # maximum_times_this_candidate_can_participate = table2[candidate_value]

    do_not_touch_table1.add(candidate_index)

    match_join_index_and_value = randomly_pick_tuple_with_value_from_table(
        table=table2,
        value=candidate_value,
        index_to_using_stats=index_to_using_stats_table_2,
    )

    if match_join_index_and_value:
        return (candidate_value, match_join_index_and_value[1])
    # else:
    #     # means that it can't be joined anymore, so let's block it
    #     do_not_touch_table1.update({index for index, stats in index_to_using_stats_table_1.items() if stats['original_value']==candidate_value})

    return None

def sample_from_full_joint(table_left, table_right, sample_size):
    join_results = [(i, j) for i in table_left for j in table_right if i == j]
    sampled = random.sample(join_results, sample_size)
    print('real join results: ', Counter([x[0] for x in sampled]))

def main():
    L1 = [2] + [1] * 100
    L2 = [1] + [2] * 200
    # sample_join_method_a(L1, L2, 1000)

    for i in range(10):
        sample_from_full_joint(L1, L2, 100)
    #sample_join_method_c(L1, L2, 50)


if __name__ == "__main__":
    main()
