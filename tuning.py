import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import create_ffnn, train_rbf_network, train_som_rbf


def tune_classifier(name, classifier, parameters, X_train, y_train, scale=True):
    # the same five folds are used for every combination and every classifier
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None
    results = []

    for params in ParameterGrid(parameters):
        scores = []
        for train_index, validation_index in cv.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_index]
            X_validation = X_train.iloc[validation_index]
            y_train_fold = y_train.iloc[train_index]
            y_validation = y_train.iloc[validation_index]

            # a fresh model for each fold. scaling is fitted on that fold only
            model = clone(classifier).set_params(**params)
            if scale:
                model = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('classifier', model),
                ])
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict(X_validation)
            scores.append(f1_score(
                y_validation, predictions, average='macro', zero_division=0,
            ))

        score = np.mean(scores)
        std = np.std(scores)
        results.append({
            'model': name, 'parameters': str(params),
            'mean_macro_f1': score, 'std_macro_f1': std,
        })
        print(f'{name}: {params}, CV macro-F1={score:.4f}, std={std:.4f}')
        if score > best_score:
            best_score = score
            best_params = params

    # train the best configuration on all training data
    best_model = clone(classifier).set_params(**best_params)
    if scale:
        best_model = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', best_model),
        ])
    best_model.fit(X_train, y_train)
    print(f'BEST {name}: {best_params}, CV macro-F1={best_score:.4f}')
    return best_model, pd.DataFrame(results), best_params, best_score


def tune_ffnn(X_train, y_train):
    parameters = {'learning_rate': [0.001, 0.005], 'epochs': [100, 200]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None
    results = []

    for params in ParameterGrid(parameters):
        scores = []
        for train_index, validation_index in cv.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_index]
            X_validation = X_train.iloc[validation_index]
            y_train_fold = y_train.iloc[train_index]
            y_validation = y_train.iloc[validation_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_validation_scaled = scaler.transform(X_validation)
            tf.keras.backend.clear_session()
            model = create_ffnn(
                [X_train_scaled.shape[1]], len(np.unique(y_train_fold)),
                learning_rate=params['learning_rate'],
            )
            model.fit(X_train_scaled, y_train_fold, epochs=params['epochs'], verbose=0)
            predictions = model.predict(X_validation_scaled, verbose=0).argmax(axis=1)

            scores.append(f1_score(
                y_validation, predictions, average='macro', zero_division=0,
            ))

        score = np.mean(scores)
        std = np.std(scores)
        results.append({
            'model': 'FFNN', 'parameters': str(params),
            'mean_macro_f1': score, 'std_macro_f1': std,
        })
        print(f'FFNN: {params}, CV macro-F1={score:.4f}, std={std:.4f}')
        if score > best_score:
            best_score = score
            best_params = params

    print(f'BEST FFNN: {best_params}, CV macro-F1={best_score:.4f}')
    return best_params, pd.DataFrame(results), best_score


def tune_rbf_network(X_train, y_train):
    parameters = {'n_centers': [3, 10, 25], 'gamma': [0.1, 0.2], 'C': [1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None
    results = []

    for params in ParameterGrid(parameters):
        scores = []
        for train_index, validation_index in cv.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_index]
            X_validation = X_train.iloc[validation_index]
            y_train_fold = y_train.iloc[train_index]
            y_validation = y_train.iloc[validation_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_validation_scaled = scaler.transform(X_validation)
            _, predictions = train_rbf_network(
                X_train_scaled, X_validation_scaled, y_train_fold,
                n_centers=params['n_centers'], gamma=params['gamma'], C=params['C'],
            )

            scores.append(f1_score(
                y_validation, predictions, average='macro', zero_division=0,
            ))

        score = np.mean(scores)
        std = np.std(scores)
        results.append({
            'model': 'RBF network', 'parameters': str(params),
            'mean_macro_f1': score, 'std_macro_f1': std,
        })
        print(f'RBF network: {params}, CV macro-F1={score:.4f}, std={std:.4f}')
        if score > best_score:
            best_score = score
            best_params = params

    print(f'BEST RBF network: {best_params}, CV macro-F1={best_score:.4f}')
    return best_params, pd.DataFrame(results), best_score


def tune_som_rbf(X_train, y_train):
    parameters = {'map_size': [(3, 3), (5, 5)], 'gamma': [0.1, 0.2], 'C': [1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None
    results = []

    for params in ParameterGrid(parameters):
        scores = []
        for train_index, validation_index in cv.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_index]
            X_validation = X_train.iloc[validation_index]
            y_train_fold = y_train.iloc[train_index]
            y_validation = y_train.iloc[validation_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_validation_scaled = scaler.transform(X_validation)
            _, _, predictions = train_som_rbf(
                X_train_scaled, X_validation_scaled, y_train_fold,
                map_size=params['map_size'], gamma=params['gamma'], C=params['C'],
            )

            scores.append(f1_score(
                y_validation, predictions, average='macro', zero_division=0,
            ))

        score = np.mean(scores)
        std = np.std(scores)
        results.append({
            'model': 'SOM + RBF', 'parameters': str(params),
            'mean_macro_f1': score, 'std_macro_f1': std,
        })
        print(f'SOM + RBF: {params}, CV macro-F1={score:.4f}, std={std:.4f}')
        if score > best_score:
            best_score = score
            best_params = params

    print(f'BEST SOM + RBF: {best_params}, CV macro-F1={best_score:.4f}')
    return best_params, pd.DataFrame(results), best_score
