import math
import os
import ast
from itertools import zip_longest

import pandas as pd
import matplotlib.pyplot as plt
from btrees.common import SAMPLING_TESTS_CSV

PLOTS_PER_LINE = 3


def get_samples_csv():
    if os.path.isfile(SAMPLING_TESTS_CSV):
        return pd.read_csv(SAMPLING_TESTS_CSV)
    return pd.DataFrame()


def dataframe_to_histogram(df):
    group_to_sizes = _unpickle_df_sampled_values_counter(df)

    groups_to_percents = list(map(_counter_to_percent, group_to_sizes))

    for bunch_of_rows in grouped(zip(groups_to_percents, df.iterrows()), PLOTS_PER_LINE):
        _dataframe_to_histogram_every_3_subplots(bunch_of_rows)



def _dataframe_to_histogram_every_3_subplots(groups_to_percents_and_df):
    assert len(groups_to_percents_and_df) <= PLOTS_PER_LINE

    fig, axes = plt.subplots(nrows=1, ncols=PLOTS_PER_LINE, figsize=(16, 5))
    fig.tight_layout(pad=3.0)
    for i, (group_to_percent, related_row) in enumerate(filter(None,groups_to_percents_and_df)):
        if not group_to_percent:
            continue
        curr_ax = axes[i]
        related_row = related_row[1].to_dict()
        pd.DataFrame([group_to_percent]).plot(kind='bar', ax=curr_ax)
        subplt_txt = f"{related_row['sample_size']}/{related_row['btree_size']} \n{related_row['name']}"
        subplt_info = f"btree_height={related_row['btree_height']}, max_leaf_size={related_row['max_leaf_size']}"
        curr_ax.set_title(subplt_info, fontsize=10, fontweight='bold')
        curr_ax.text(0, -0.08, s=subplt_txt, size=12, ha="center", fontstyle="normal", family="sans-serif")
        curr_ax.text(0, -0.08, s=subplt_txt, size=12, ha="center", fontstyle="normal", family="sans-serif")
        for p in curr_ax.patches:
            curr_ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip_longest(*[iter(iterable)]*n)

def _counter_to_percent(group_to_size):
    data_length = sum(group_to_size.values())
    return {key: occurences / data_length for key, occurences in group_to_size.items()}

def _unpickle_df_sampled_values_counter(df):
    sampled_values_counter_strs = df["sampled_values_counter"]
    sampled_values_counter_lists = map(ast.literal_eval, sampled_values_counter_strs)
    group_to_sizes = list(map(dict, sampled_values_counter_lists))
    return group_to_sizes