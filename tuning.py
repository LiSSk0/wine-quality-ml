import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def tune_knn(X_train, y_train):
    # LabML6_SOM.py
    # Сохранена структура ручного перебора; параметры заменены на параметры KNN.
    K_values = [3, 5, 10, 15]
    weights_values = ['uniform', 'distance']

    best_score = -np.inf
    best_K = None
    best_weights = None
    best_std = None
    results = []

    # Добавлено для CV: одинаковые пять фолдов для всех комбинаций.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(X_train, y_train))

    for K in K_values:
        for weights in weights_values:
            scores = []

            for train_index, validation_index in folds:
                X_train_fold = X_train.iloc[train_index]
                X_validation = X_train.iloc[validation_index]
                y_train_fold = y_train.iloc[train_index]
                y_validation = y_train.iloc[validation_index]

                # ML_DL_Data_Cleaning_One_Hot_Encoding_Guide.docx
                # LabML9_KNN.py
                clf = Pipeline(steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(
                        n_neighbors=K, weights=weights,
                    )),
                ])
                clf.fit(X_train_fold, y_train_fold)
                predictions = clf.predict(X_validation)

                # Добавлено для CV: метрика классификации вместо ошибки SOM.
                score = f1_score(
                    y_validation, predictions, average='macro', zero_division=0,
                )
                scores.append(score)

            score = np.mean(scores)
            std = np.std(scores)
            results.append({
                'K': K,
                'weights': weights,
                'mean_macro_f1': score,
                'std_macro_f1': std,
                **{f'fold_{i + 1}_macro_f1': value for i, value in enumerate(scores)},
            })

            # LabML6_SOM.py
            print(f"K={K}, weights={weights}, score={score:.4f}, std={std:.4f}")

            # Для F1 выбираем максимум; в примере SOM выбирался минимум ошибки.
            if score > best_score:
                best_score = score
                best_K = K
                best_weights = weights
                best_std = std

    print("\nBEST PARAMETERS")
    print("K =", best_K)
    print("weights =", best_weights)
    print("CV macro-F1 =", best_score)
    print("CV standard deviation =", best_std)

    # LabML9_KNN.py
    # Лучшая конфигурация заново обучается на всей обучающей выборке.
    best_clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=best_K, weights=best_weights,
        )),
    ])
    best_clf.fit(X_train, y_train)
    return best_clf, pd.DataFrame(results)
