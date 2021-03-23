import os

import pandas as pd
from btrees.common import SAMPLING_TESTS_CSV


def get_samples_csv():
    if os.path.isfile(SAMPLING_TESTS_CSV):
        return pd.read_csv(SAMPLING_TESTS_CSV)
    return pd.DataFrame()