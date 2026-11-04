from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd


def preprocessing(df: pd.DataFrame):
    X = df.drop(columns="quality")
    y = df["quality"].map({"low": 0, "medium": 1, "high": 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    # numeric_pipeline = Pipeline(steps=[
    #     ("imputer", SimpleImputer(strategy="median")),
    #     ("scaler", StandardScaler())  # 0 - mean value
    # ])

    # categorical_pipeline = Pipeline(steps=[
    #     ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    #     ("onehot", OneHotEncoder(handle_unknown="ignore"))
    # ])

    # applying numeric_pipeline to X
    # preprocessor = ColumnTransformer(transformers=[
    #     ("num", numeric_pipeline, X.columns.tolist())
    #     ("cat", categorical_pipeline, categorical_cols)
    # ])

    model = Pipeline(steps=[
        # ("preprocess", preprocessor),
        ("scaler", StandardScaler()),  # 0 - mean value
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    print("Test accuracy:", model.score(X_test, y_test))
    print(model.named_steps["scaler"].get_feature_names_out())
