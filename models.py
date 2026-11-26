from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from sklearn_som.som import SOM


# LabML3_FF.py
def create_ffnn(input_shape, Nbclasses, learning_rate=0.001):
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
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return clf


# LabML7_RBF.py
def train_rbf_network(X_train, X_test, y_train, n_centers=None, gamma=0.2, C=1):
    if n_centers is None:
        n_centers = len(set(y_train))
    kmeans = KMeans(n_clusters=n_centers, random_state=42)
    kmeans.fit(X_train)
    centers = kmeans.cluster_centers_
    # centers передаются из обучающих данных, без использования test.
    Ztrain = rbf_kernel(X_train, centers, gamma=gamma)
    clf = svm.SVC(
        C=C,
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
def train_som_rbf(X_train, X_test, y_train, map_size=(5, 5), gamma=0.1, C=1):
    m1, n1 = map_size
    dim = X_train.shape[1]
    som = SOM(
        m=m1, n=n1, dim=dim, random_state=42,
        max_iter=50000, lr=1, sigma=1,
    )
    som.fit(X_train)
    centers = som.weights
    Z = rbf_kernel(X_train, centers, gamma=gamma)
    clf = svm.SVC(
        C=C,
        kernel='linear',
        probability=True,
        tol=1e-3,
        random_state=42,
    )
    clf.fit(Z, y_train)
    Z1 = rbf_kernel(X_test, centers, gamma=gamma)
    predictions = clf.predict(Z1)
    return som, clf, predictions
