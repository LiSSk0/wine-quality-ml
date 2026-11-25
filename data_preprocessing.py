from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

from default_models import (
    create_default_decision_tree,
    create_default_knn,
    create_default_linear_classifier,
    create_default_linear_svm,
    create_default_logistic_regression,
    create_default_mlp,
    create_default_perceptron,
    create_default_random_forest,
    create_default_rbf_svm,
    create_dummy_classifier,
    create_initial_ffnn,
    train_initial_rbf,
    train_initial_som_rbf,
)

from plots import plot_confusion_matrix


SHOW_PLOTS = False
PLOTS_DIR = Path(__file__).resolve().parent / "results" / "plots" / "baseline"


def preprocessing(df: pd.DataFrame):
    X = df.drop(columns="quality")
    y = df["quality"].map({"low": 0, "medium": 1, "high": 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    print("\nBaseline: default library parameters, random_state=42")

    # LogisticRegression prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),  # 0 - mean value
        ("classifier", create_default_logistic_regression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLogisticRegression")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="LogisticRegression",
        output_path=PLOTS_DIR / "logistic_regression.png", show=SHOW_PLOTS,
    )

    # Perceptron prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_perceptron())
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nPerceptron")
    score = model.score(X_test, y_test)
    print("Accuracy:", score)

    plot_confusion_matrix(
        y_test, predictions, title="Perceptron",
        output_path=PLOTS_DIR / "perceptron.png", show=SHOW_PLOTS,
    )

    # MLP prediction
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_mlp())
    ])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("\nMLP")
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

    plot_confusion_matrix(
        y_test, predictions, title="MLP",
        output_path=PLOTS_DIR / "mlp.png", show=SHOW_PLOTS,
    )

    # Linear classifier prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_linear_classifier())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLinear classifier (SGD)")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="Linear classifier (SGD)",
        output_path=PLOTS_DIR / "linear_sgd.png", show=SHOW_PLOTS,
    )

    # Linear SVM prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_linear_svm())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLinear SVM")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="Linear SVM",
        output_path=PLOTS_DIR / "linear_svm.png", show=SHOW_PLOTS,
    )

    # RBF SVM prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_rbf_svm())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nRBF SVM")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="RBF SVM",
        output_path=PLOTS_DIR / "rbf_svm.png", show=SHOW_PLOTS,
    )

    # KNN prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_knn())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nKNN (default, K=5)")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="KNN (default, K=5)",
        output_path=PLOTS_DIR / "knn.png", show=SHOW_PLOTS,
    )

    # Dummy majority baseline
    model = create_dummy_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDummy (most frequent)")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="Dummy (most frequent)",
        output_path=PLOTS_DIR / "dummy.png", show=SHOW_PLOTS,
    )

    # Decision Tree prediction
    model = create_default_decision_tree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDecision Tree")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="Decision Tree",
        output_path=PLOTS_DIR / "decision_tree.png", show=SHOW_PLOTS,
    )

    # Random Forest prediction
    model = create_default_random_forest()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nRandom Forest")
    print("Accuracy:", model.score(X_test, y_test))

    plot_confusion_matrix(
        y_test, predictions, title="Random Forest",
        output_path=PLOTS_DIR / "random_forest.png", show=SHOW_PLOTS,
    )

    print("\nBaseline: initial lab configurations (not library defaults)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LabML3_FF.py
    clf = create_initial_ffnn([X_train_scaled.shape[1]], len(np.unique(y_train)))
    clf.fit(X_train_scaled, y_train, epochs=500, verbose=0)
    pred = clf.predict(X_test_scaled, verbose=0)
    predictions = pred.argmax(axis=1)
    print("\nFFNN (initial lab configuration)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plot_confusion_matrix(
        y_test, predictions, title="FFNN (initial lab configuration)",
        output_path=PLOTS_DIR / "ffnn.png", show=SHOW_PLOTS,
    )

    # LabML7_RBF.py
    clf, predictions = train_initial_rbf(
        X_train_scaled, X_test_scaled, y_train,
    )
    print("\nRBF network (initial configuration)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plot_confusion_matrix(
        y_test, predictions, title="RBF network (initial configuration)",
        output_path=PLOTS_DIR / "rbf_network.png", show=SHOW_PLOTS,
    )

    # LabML8_SOM_RBF.py
    som, clf, predictions = train_initial_som_rbf(
        X_train_scaled, X_test_scaled, y_train,
    )
    print("\nSOM + RBF (initial lab configuration)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plot_confusion_matrix(
        y_test, predictions, title="SOM + RBF (initial lab configuration)",
        output_path=PLOTS_DIR / "som_rbf.png", show=SHOW_PLOTS,
    )
