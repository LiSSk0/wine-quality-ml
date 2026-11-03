import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


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

# Find all missing values:
df = df.replace(r"^\s*$", pd.NA, regex=True)
cols = df.columns.tolist()  # all columns are numerical
df = df.apply(pd.to_numeric, errors="coerce")  # invalid numbers turn into missing value
print(df.isna().sum())

# Delete duplicates:
df = df.drop_duplicates()

# Trying to inspect outliers
print(df.describe(include="all").to_string())


# Divide quality into 3 classes
def group_quality(q):
    if q <= 5: return "low"
    elif q == 6: return "medium"
    else: return "high"


df["quality"] = df["quality"].apply(group_quality)
print(df["quality"].isna().sum())
print(df["quality"].value_counts())

# Divide data into train/test



