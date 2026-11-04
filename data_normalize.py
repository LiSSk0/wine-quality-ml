import pandas as pd


def normalize(df: pd.DataFrame):
    # Find all missing values:
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    # cols = df.columns.tolist()  # all columns are numerical
    df = df.apply(pd.to_numeric, errors="coerce")  # invalid numbers turn into missing value
    print("missing values:", df.isna().sum())

    df = df.drop_duplicates()

    # to see outliers
    print(df.describe(include="all").to_string())

    # Divide quality into 3 classes
    def group_quality(q):
        if q <= 5:
            return "low"
        elif q == 6:
            return "medium"
        else:
            return "high"

    df["quality"] = df["quality"].apply(group_quality)
    print(df["quality"].isna().sum())
    print(df["quality"].value_counts())

    return df
