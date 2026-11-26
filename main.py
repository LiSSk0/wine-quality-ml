import pandas as pd
from data_normalize import normalize
from data_preprocessing import preprocessing
import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def main():
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv('redwine_data.csv', sep=';')

    print(df.head())
    # print(df.info())
    # print(df.dtypes)

    df = normalize(df)

    # Divide data into train/test
    preprocessing(df)


if __name__ == '__main__':
    main()
