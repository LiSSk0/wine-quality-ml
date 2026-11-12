import pandas as pd
from scipy.odr import Model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


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
        ("classifier", LogisticRegression(max_iter=1000))
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
        ("classifier", Perceptron(
            tol=1e-15,  # training stops when the loss improvement falls below this value
            eta0=0.01,  # initial learning rate for weight updates
            max_iter=10000,  # maximum number of passes over the training data
            random_state=None  # no fixed seed, so results vary between runs
        ))
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

    # FF prediction
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            solver="lbfgs",  # quasi-Newton optimizer used for training (good for small datasets)
            hidden_layer_sizes=(30, 12, 15, 10, 19),  # 5 hidden layers with 30, 12, 15, 10, and 19 neurons respectively
            activation="tanh",  # hyperbolic tangent activation function for hidden neurons
            alpha=1e-5,  # L2 regularization strength (very small, so little regularization)
            max_iter=1000,  # maximum number of training iterations
            random_state=None  # no fixed seed
        ))
    ])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("\nFF")
    # print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

    plt.figure()
    plot_confusion_matrix(y_test, predictions)
    plt.title("FF")
    plt.show()
