import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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


def plot_confusion_matrix(y_true, y_pred, cmap="coolwarm", subplot_position=111):
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Créer la figure et spécifier le subplot
    plt.subplot(subplot_position)

    # Afficher la matrice sous forme de heatmap
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im)  # Ajouter une barre de couleur

    # Ajouter les valeurs dans les cases
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=12, fontweight="bold")

    # Ajouter les labels des axes
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(3), ["low", "medium", "high"])
    plt.yticks(np.arange(3), ["low", "medium", "high"])

    # Ajuster l'affichage
    plt.tight_layout()


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

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("LogisticRegression")
    plt.show()

    # Perceptron prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_perceptron())
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nPerceptron")
    # print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    score = model.score(X_test, y_test)
    print("Accuracy:", score)

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Perceptron")
    plt.show()

    # MLP prediction
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_mlp())
    ])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("\nMLP")
    # print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("MLP")
    plt.show()

    # Linear classifier prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_linear_classifier())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLinear classifier (SGD)")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Linear classifier (SGD)")
    plt.show()

    # Linear SVM prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_linear_svm())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLinear SVM")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Linear SVM")
    plt.show()

    # RBF SVM prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_rbf_svm())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nRBF SVM")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("RBF SVM")
    plt.show()

    # KNN prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_default_knn())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nKNN (default, K=5)")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("KNN (default, K=5)")
    plt.show()

    # Dummy majority baseline
    model = create_dummy_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDummy (most frequent)")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Dummy (most frequent)")
    plt.show()

    # Decision Tree prediction
    model = create_default_decision_tree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDecision Tree")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Decision Tree")
    plt.show()

    # Random Forest prediction
    model = create_default_random_forest()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nRandom Forest")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Random Forest")
    plt.show()

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

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("FFNN (initial lab configuration)")
    plt.show()

    # LabML7_RBF.py
    clf, predictions = train_initial_rbf(
        X_train_scaled, X_test_scaled, y_train,
    )
    print("\nRBF network (initial configuration)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("RBF network (initial configuration)")
    plt.show()

    # LabML8_SOM_RBF.py
    som, clf, predictions = train_initial_som_rbf(
        X_train_scaled, X_test_scaled, y_train,
    )
    print("\nSOM + RBF (initial lab configuration)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("SOM + RBF (initial lab configuration)")
    plt.show()
