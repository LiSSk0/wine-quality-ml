import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from models import (
    create_decision_tree,
    create_ffnn,
    create_knn,
    create_knn_sqrt,
    create_linear_classifier,
    create_linear_svm,
    create_logistic_regression,
    create_mlp,
    create_perceptron,
    create_random_forest,
    create_rbf_svm,
    train_rbf,
    train_som_rbf,
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

    # LogisticRegression prediction
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),  # 0 - mean value
        ("classifier", create_logistic_regression())
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
        ("classifier", create_perceptron())
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
        ("classifier", create_mlp())
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
        ("classifier", create_linear_classifier())
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
        ("classifier", create_linear_svm())
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
        ("classifier", create_rbf_svm())
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
        ("classifier", create_knn())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nKNN (K=10)")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("KNN (K=10)")
    plt.show()

    # KNN prediction with K from the training sample count
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_knn_sqrt(len(X_train)))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nKNN (K=sqrt(n_train))")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("KNN (K=sqrt(n_train))")
    plt.show()

    # Decision Tree prediction
    model = create_decision_tree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDecision Tree")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Decision Tree")
    plt.show()

    # Random Forest prediction
    model = create_random_forest()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nRandom Forest")
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("Random Forest")
    plt.show()

    # LabML3_FF.py
    # LabML7_RBF.py
    # LabML8_SOM_RBF.py
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # FFNN prediction
    clf = create_ffnn([X_train_scaled.shape[1]], len(np.unique(y_train)))
    clf.fit(X_train_scaled, y_train, epochs=500, verbose=0)
    pred = clf.predict(X_test_scaled, verbose=0)
    predictions = pred.argmax(axis=1)
    print("\nFFNN (TensorFlow)")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("FFNN (TensorFlow)")
    plt.show()

    # LabML5_Kmeans.py
    nb_clusters = len(np.unique(y_train))
    model = KMeans(n_clusters=nb_clusters)
    model.fit(X_train_scaled)
    centers = model.cluster_centers_

    # RBF network prediction
    clf, predictions = train_rbf(
        X_train_scaled, X_test_scaled, y_train, centers,
    )
    print("\nRBF network")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("RBF network")
    plt.show()

    # SOM + RBF prediction
    som, clf, predictions = train_som_rbf(
        X_train_scaled, X_test_scaled, y_train,
    )
    print("\nSOM + RBF")
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("SOM + RBF")
    plt.show()
