import pandas as pd
from scipy.odr import Model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def preprocessing(df: pd.DataFrame):
    X = df.drop(columns="quality")
    y = df["quality"].map({"low": 0, "medium": 1, "high": 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", Perceptron(
            tol=1e-15,
            eta0=0.01,
            max_iter=10000,
            random_state=None
        ))
    ])

    #model = Pipeline(steps=[
        # ("preprocess", preprocessor),
    #    ("scaler", StandardScaler()),  # 0 - mean value
    #    ("classifier", LogisticRegression(max_iter=1000))
    #])


    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nPerceptron")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    score = model.score(X_test, y_test)
    print("Accuracy:", score)

    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            solver="lbfgs",
            hidden_layer_sizes=(30, 12, 15, 10, 19),
            activation="tanh",
            alpha=1e-5,
            max_iter=1000,
            random_state=None
        ))
    ])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("\nFF")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)
