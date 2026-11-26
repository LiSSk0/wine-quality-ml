import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
from data_normalize import normalize
from data_preprocessing import preprocessing


def main():
    df = pd.read_csv('redwine_data.csv', sep=';')
    print(df.head())
    df = normalize(df)

    # Run baseline, tuning and comparison in order.
    preprocessing(df)


if __name__ == '__main__':
    main()
