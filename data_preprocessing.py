import os
import numpy as np
import pandas as pd
from sklearn import svm, tree
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import create_ffnn, train_rbf_network, train_som_rbf
from plots import plot_confusion_matrix
from tuning import tune_classifier, tune_ffnn, tune_rbf_network, tune_som_rbf


SHOW_PLOTS = False


def record_result(name, version, y_test, predictions, results, reports, parameters=None, cv_score=None):
    # print and remember test metrics and class reports
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    print(f'\n{name} ({version})')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro-F1: {macro_f1:.4f}')
    print(classification_report(
        y_test, predictions, labels=[0, 1, 2],
        target_names=['low', 'medium', 'high'], digits=4, zero_division=0,
    ))
    report = classification_report(
        y_test, predictions, labels=[0, 1, 2],
        target_names=['low', 'medium', 'high'], output_dict=True, zero_division=0,
    )
    for label, values in report.items():
        if label == 'accuracy':
            values = {'accuracy': values, 'support': len(y_test)}
        reports.append({'model': name, 'version': version, 'class': label, **values})
    results.append({
        'model': name, 'version': version, 'accuracy': accuracy,
        'macro_f1': macro_f1, 'parameters': str(parameters), 'cv_macro_f1': cv_score,
    })
    filename = name.lower().replace(' ', '_').replace('+', 'plus')
    plot_confusion_matrix(
        y_test, predictions, title=f'{name} ({version})',
        output_path=f'results/plots/{version}/{filename}.png', show=SHOW_PLOTS,
    )


def preprocessing(df: pd.DataFrame):
    X = df.drop(columns='quality')
    y = df['quality'].map({'low': 0, 'medium': 1, 'high': 2})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y,
    )
    results = []
    reports = []
    cv_results = []
    os.makedirs('results', exist_ok=True)

    print('\nBASELINE: default parameters')

    # Logistic Regression
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Logistic Regression', 'baseline', y_test, predictions, results, reports)

    # Perceptron
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', Perceptron(random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Perceptron', 'baseline', y_test, predictions, results, reports)

    # MLP
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('MLP', 'baseline', y_test, predictions, results, reports)

    # Linear SGD
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Linear SGD', 'baseline', y_test, predictions, results, reports)

    # Linear SVM
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', svm.SVC(kernel='linear', random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Linear SVM', 'baseline', y_test, predictions, results, reports)

    # RBF SVM
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', svm.SVC(random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('RBF SVM', 'baseline', y_test, predictions, results, reports)

    # KNN
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('KNN', 'baseline', y_test, predictions, results, reports)

    # Decision Tree
    model = tree.DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Decision Tree', 'baseline', y_test, predictions, results, reports)

    # Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Random Forest', 'baseline', y_test, predictions, results, reports)

    # Dummy: comparison with always predicting the most frequent class
    model = DummyClassifier(strategy='most_frequent', random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    record_result('Dummy', 'baseline', y_test, predictions, results, reports)

    print('\nBASELINE: initial lab configurations')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # FFNN
    model = create_ffnn([X_train_scaled.shape[1]], len(np.unique(y_train)))
    model.fit(X_train_scaled, y_train, epochs=500, verbose=0)
    predictions = model.predict(X_test_scaled, verbose=0).argmax(axis=1)
    record_result('FFNN', 'baseline', y_test, predictions, results, reports,
                  parameters={'epochs': 500, 'learning_rate': 0.001})

    # RBF network: centers are selected using training data only
    _, predictions = train_rbf_network(X_train_scaled, X_test_scaled, y_train)
    record_result('RBF network', 'baseline', y_test, predictions, results, reports,
                  parameters={'n_centers': 3, 'gamma': 0.2, 'C': 1})

    # SOM + RBF
    _, _, predictions = train_som_rbf(X_train_scaled, X_test_scaled, y_train)
    record_result('SOM + RBF', 'baseline', y_test, predictions, results, reports,
                  parameters={'map_size': (5, 5), 'gamma': 0.1, 'C': 1})

    # tune on train first. evaluate on test only after the best parameters are chosen
    print('\nTUNING: five-fold CV on train, selection by macro-F1')

    # Logistic Regression
    model, table, params, score = tune_classifier(
        'Logistic Regression', LogisticRegression(random_state=42),
        {'C': [0.1, 1, 10], 'class_weight': [None, 'balanced']},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Logistic Regression', 'tuned', y_test, predictions, results, reports, params, score)

    # Perceptron
    model, table, params, score = tune_classifier(
        'Perceptron', Perceptron(random_state=42),
        {'eta0': [0.01, 0.1, 1], 'penalty': [None, 'l2']},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Perceptron', 'tuned', y_test, predictions, results, reports, params, score)

    # MLP
    model, table, params, score = tune_classifier(
        'MLP', MLPClassifier(random_state=42),
        {'hidden_layer_sizes': [(30,), (50,), (100,)], 'alpha': [0.0001, 0.01], 'max_iter': [1000]},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('MLP', 'tuned', y_test, predictions, results, reports, params, score)

    # Linear SGD
    model, table, params, score = tune_classifier(
        'Linear SGD', SGDClassifier(random_state=42),
        {'loss': ['hinge', 'log_loss'], 'alpha': [0.0001, 0.001, 0.01]},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Linear SGD', 'tuned', y_test, predictions, results, reports, params, score)

    # Linear SVM
    model, table, params, score = tune_classifier(
        'Linear SVM', svm.SVC(kernel='linear', random_state=42),
        {'C': [0.1, 1, 10], 'class_weight': [None, 'balanced']},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Linear SVM', 'tuned', y_test, predictions, results, reports, params, score)

    # RBF SVM
    model, table, params, score = tune_classifier(
        'RBF SVM', svm.SVC(random_state=42),
        {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'class_weight': [None, 'balanced']},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('RBF SVM', 'tuned', y_test, predictions, results, reports, params, score)

    # KNN
    model, table, params, score = tune_classifier(
        'KNN', KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 10, 15], 'weights': ['uniform', 'distance']},
        X_train, y_train,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('KNN', 'tuned', y_test, predictions, results, reports, params, score)

    # Decision Tree
    model, table, params, score = tune_classifier(
        'Decision Tree', tree.DecisionTreeClassifier(random_state=42),
        {'criterion': ['gini', 'entropy'], 'max_depth': [None, 4, 8], 'min_samples_leaf': [1, 5]},
        X_train, y_train, scale=False,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Decision Tree', 'tuned', y_test, predictions, results, reports, params, score)

    # Random Forest
    model, table, params, score = tune_classifier(
        'Random Forest', RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 6], 'min_samples_leaf': [1, 3]},
        X_train, y_train, scale=False,
    )
    cv_results.append(table)
    predictions = model.predict(X_test)
    record_result('Random Forest', 'tuned', y_test, predictions, results, reports, params, score)

    # RBF network
    params, table, score = tune_rbf_network(X_train, y_train)
    cv_results.append(table)
    _, predictions = train_rbf_network(
        X_train_scaled, X_test_scaled, y_train,
        n_centers=params['n_centers'], gamma=params['gamma'], C=params['C'],
    )
    record_result('RBF network', 'tuned', y_test, predictions, results, reports, params, score)

    # SOM + RBF
    params, table, score = tune_som_rbf(X_train, y_train)
    cv_results.append(table)
    _, _, predictions = train_som_rbf(
        X_train_scaled, X_test_scaled, y_train,
        map_size=params['map_size'], gamma=params['gamma'], C=params['C'],
    )
    record_result('SOM + RBF', 'tuned', y_test, predictions, results, reports, params, score)

    # FFNN
    params, table, score = tune_ffnn(X_train, y_train)
    cv_results.append(table)
    tf_model = create_ffnn(
        [X_train_scaled.shape[1]], len(np.unique(y_train)),
        learning_rate=params['learning_rate'],
    )
    tf_model.fit(X_train_scaled, y_train, epochs=params['epochs'], verbose=0)
    predictions = tf_model.predict(X_test_scaled, verbose=0).argmax(axis=1)
    record_result('FFNN', 'tuned', y_test, predictions, results, reports, params, score)

    # Save the results needed for the report.
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/model_results.csv', index=False)
    pd.DataFrame(reports).to_csv('results/classification_reports.csv', index=False)
    pd.concat(cv_results, ignore_index=True).to_csv('results/cv_results.csv', index=False)

    baseline = results_df[results_df['version'] == 'baseline'][
        ['model', 'accuracy', 'macro_f1']
    ].rename(columns={'accuracy': 'baseline_accuracy', 'macro_f1': 'baseline_macro_f1'})
    tuned = results_df[results_df['version'] == 'tuned'][
        ['model', 'accuracy', 'macro_f1', 'parameters', 'cv_macro_f1']
    ].rename(columns={'accuracy': 'tuned_accuracy', 'macro_f1': 'tuned_macro_f1'})
    comparison = baseline.merge(tuned, on='model', how='left')
    comparison.to_csv('results/comparison.csv', index=False)
    print('\nCOMPARISON ON THE SAME TEST SET')
    print(comparison[[
        'model', 'baseline_accuracy', 'tuned_accuracy',
        'baseline_macro_f1', 'tuned_macro_f1',
    ]].round(4).to_string(index=False, na_rep='-'))
