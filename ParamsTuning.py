from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np


def mlp_param_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
    parameters = [{
        'hidden_layer_sizes': [(100, 50, 25), (100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [.1, .01, 10 ** -3, 10 ** -4],
        'learning_rate': ['constant', 'adaptive'],
    }]

    clf = model_selection.GridSearchCV(MLPClassifier(max_iter=10000), param_grid=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_


def svm_param_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
    parameters = [{"kernel": ['rbf'], 'C': [0.1, 1, 10, 25, 50, 75, 100],
                   "gamma": [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10, 10 ** 2, 10 ** 3, 10 ** 4],
                   "decision_function_shape": ["ovo", "ovr"]
                   },
                  {"kernel": ['linear'], "C": [0.1, 1, 10], "decision_function_shape": ["ovo", "ovr"]}]
    clf = model_selection.GridSearchCV(SVC(), param_grid=parameters, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_


def random_forest_param_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
    param_grid ={'criterion': ['entropy', 'gini'],
               'max_depth': list(np.linspace(10, 150, 10, dtype = int)),
               'max_features': ['auto', 'sqrt','log2', None],
               'min_samples_leaf': [4, 6, 8, 12],
               'min_samples_split': [5, 7, 10, 14],
               'n_estimators': list(np.linspace(150, 300, 10, dtype = int))}

    clf = model_selection.GridSearchCV(RandomForestClassifier(),  param_grid=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf


def sgd_param_selection(X, y, n_folds, metric):
    param_grid = {
        "loss": ["hinge", "log", "squared_hinge", "modified_huber"],
        'max_iter': [1000],
        'l1_ratio': [0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2],
        "penalty": ["l2", "l1", 'elasticnet'],
    }
    clf = model_selection.GridSearchCV(SGDClassifier(max_iter=6000), param_grid=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf


def knn_param_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
    param_grid = {
        'n_neighbors': [3, 5, 7, 11],
        'metric': ["minkowski", "euclidean", "chebyshev"],
        "p": [3,4,5]
    }

    clf = model_selection.GridSearchCV(KNeighborsClassifier(),param_grid=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf
