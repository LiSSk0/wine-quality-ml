import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from data_normalize import normalize
from data_preprocessing import preprocessing


def main():
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv('redwine_data.csv', sep=';')
    """
    df = pd.DataFrame({
        "Age": [20, np.nan, 22, 21, 24, 20],
        "Hours": [5, 7, "", 3, 8, 6],
        "City": ["Tripoli", "Beirut", "Tripoli", " ", "Beirut", "Saida"],
        "Program": ["Engineering", "Science", "Engineering", "Medicine", "", "Science"],
        "Passed": ["Yes", "No", "Yes", "No", "Yes", "No"]
    })
    """

    print(df.head())
    # print(df.info())
    # print(df.dtypes)
    print("Duplicates:", df.duplicated().sum(), '\n')

    df = normalize(df)

    # Divide data into train/test
    preprocessing(df)


if __name__ == '__main__':
    main()
