from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from sklearn_som.som import SOM


# LabML3_FF.py
def create_ffnn(input_shape, Nbclasses):
    tf.keras.utils.set_random_seed(42)

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


# LabML7_RBF.py
def train_rbf(X_train, X_test, y_train, centers, gamma=0.2):
    # centers передаются из обучающих данных, без использования test.
    Ztrain = rbf_kernel(X_train, centers, gamma=gamma)
    clf = svm.SVC(
        C=1,
        kernel='linear',
        probability=True,
        tol=1e-3,
        random_state=42,
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
        m=m1, n=n1, dim=dim, random_state=42,
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
        random_state=42,
    )
    clf.fit(Z, y_train)
    Z1 = rbf_kernel(X_test, centers, gamma=gamma)
    predictions = clf.predict(Z1)
    return som, clf, predictions


# LabML5_Kmeans.py
# LabML7_RBF.py
def train_initial_rbf(X_train, X_test, y_train):
    nb_clusters = len(set(y_train))
    model = KMeans(n_clusters=nb_clusters, random_state=42)
    model.fit(X_train)
    centers = model.cluster_centers_
    return train_rbf(X_train, X_test, y_train, centers, gamma=0.2)
