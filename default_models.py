from sklearn import svm, tree
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from models import create_ffnn, train_rbf, train_som_rbf


# Стандартные параметры библиотеки; seed фиксирован для повторяемости.
def create_default_logistic_regression():
    return LogisticRegression(random_state=42)


def create_default_linear_classifier():
    return SGDClassifier(random_state=42)


def create_default_perceptron():
    return Perceptron(random_state=42)


def create_default_mlp():
    return MLPClassifier(random_state=42)


def create_default_linear_svm():
    return svm.SVC(kernel='linear', random_state=42)


def create_default_rbf_svm():
    return svm.SVC(random_state=42)


def create_default_knn():
    return KNeighborsClassifier()


def create_default_decision_tree():
    return tree.DecisionTreeClassifier(random_state=42)


def create_default_random_forest():
    return RandomForestClassifier(random_state=42)


def create_dummy_classifier():
    return DummyClassifier(strategy='most_frequent', random_state=42)


# У составных моделей нет общих library defaults.
# Ниже используются исходные конфигурации из лабораторных, до CV-настройки.

# LabML3_FF.py
def create_initial_ffnn(input_shape, Nbclasses):
    return create_ffnn(input_shape, Nbclasses)


# LabML5_Kmeans.py
# LabML7_RBF.py
def train_initial_rbf(X_train, X_test, y_train):
    nb_clusters = len(set(y_train))
    model = KMeans(n_clusters=nb_clusters, random_state=42)
    model.fit(X_train)
    centers = model.cluster_centers_
    return train_rbf(X_train, X_test, y_train, centers, gamma=0.2)


# LabML8_SOM_RBF.py
def train_initial_som_rbf(X_train, X_test, y_train):
    return train_som_rbf(X_train, X_test, y_train)
