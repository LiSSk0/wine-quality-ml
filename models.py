import numpy as np
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from sklearn_som.som import SOM

# ML_DL_Data_Cleaning_One_Hot_Encoding_Guide.docx
def create_logistic_regression():
    return LogisticRegression(max_iter=1000)


# LabML1_perc_lin.py
def create_linear_classifier():
    return SGDClassifier(eta0=0.05, max_iter=1000, tol=1e-15, random_state=10)


# LabML1_perc_lin.py
def create_perceptron():
    return Perceptron(tol=1e-15, eta0=0.01, max_iter=10000, random_state=None)


# LabML2_MLP.py
def create_mlp():
    return MLPClassifier(
        solver='lbfgs',
        hidden_layer_sizes=(30, 12, 15, 10, 19),
        activation='tanh',
        alpha=1e-5,
        max_iter=1000,
        random_state=None,
    )


# LabML3_FF.py
def create_ffnn(input_shape, Nbclasses):

    clf = models.Sequential()
    clf.add(Dense(100, activation='sigmoid', input_shape=input_shape))
    clf.add(Dense(units=70, activation='sigmoid'))
    clf.add(Dense(units=50, activation='tanh'))
    clf.add(Dense(units=60, activation='linear'))
    clf.add(Dense(units=90, activation='tanh'))
    clf.add(Dense(units=40, activation='tanh'))
    clf.add(Dense(Nbclasses, activation='softmax'))

    # На выходе softmax: loss получает вероятности, а не logits.
    clf.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return clf


# LabML4_SVM.py
def create_linear_svm():
    return svm.SVC(kernel='linear')


# LabML4_SVM.py
def create_rbf_svm():
    return svm.SVC(kernel='rbf', gamma=1, C=10, probability=True)


# LabML7_RBF.py
def train_rbf(X_train, X_test, y_train, centers, gamma=0.2):
    # centers передаются из обучающих данных, без использования test.
    Ztrain = rbf_kernel(X_train, centers, gamma=gamma)
    clf = svm.SVC(
        C=1,
        kernel='linear',
        probability=True,
        tol=1e-3,
        random_state=None,
    )
    clf.fit(Ztrain, y_train)
    Ztest = rbf_kernel(X_test, centers, gamma=gamma)
    predictions = clf.predict(Ztest)
    return clf, predictions


# LabML8_SOM_RBF.py
def train_som_rbf(X_train, X_test, y_train):
    m1 = 5
    n1 = 5
    dim = X_train.shape[1]
    som = SOM(
        m=m1, n=n1, dim=dim, random_state=None,
        max_iter=50000, lr=1, sigma=1,
    )
    som.fit(X_train)
    centers = som.weights
    gamma = 0.1
    Z = rbf_kernel(X_train, centers, gamma=gamma)
    clf = svm.SVC(
        C=1,
        kernel='linear',
        probability=True,
        tol=1e-3,
        random_state=None,
    )
    clf.fit(Z, y_train)
    Z1 = rbf_kernel(X_test, centers, gamma=gamma)
    predictions = clf.predict(Z1)
    return som, clf, predictions


# LabML9_KNN.py
def create_knn():
    return KNeighborsClassifier(n_neighbors=10)


# LabML9_KNN.py
def create_knn_sqrt(n_samples):
    K = int(np.sqrt(n_samples))
    return KNeighborsClassifier(n_neighbors=K)


# Part 3 - Decisiontree .pptx
def create_decision_tree():
    return tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=4, splitter='best',
    )


# Part 4 - RandomForest.pptx
def create_random_forest():
    return RandomForestClassifier(
        bootstrap=True,
        max_features=2,
        criterion='entropy',
        n_estimators=20,
        max_depth=6,
        random_state=10,
        verbose=1,
    )
