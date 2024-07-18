import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error


SIZES = [1, 2, 3 ,4 ,5, 6, 7, 8, 9, 10, 15, 20, 35, 50, 75, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400]

REDUCTION_METHODS = ['pca', 'umap', 'tsne']

DATABASES = ['go', 'kegg', 'msigdb']
ALL_DATABASES = 'all'

CLASSIFIERS = {
    'Reg': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'SVM': SVC,
    'DTree': DecisionTreeClassifier,
    'RF': RandomForestClassifier,
    'LGBM': LGBMClassifier,
    'XGB': XGBClassifier,
    'GradBoost': GradientBoostingClassifier,
    'MLP': MLPClassifier
}
REGRESSORS = {
    'Reg': LinearRegression,
    'KNN': KNeighborsRegressor,
    'SVM': SVR,
    'DTree': DecisionTreeRegressor,
    'RF': RandomForestRegressor,
    'LGBM': LGBMRegressor,
    'XGB': XGBRegressor,
    'GradBoost': GradientBoostingRegressor,
    'MLP': MLPRegressor
}
assert CLASSIFIERS.keys() == REGRESSORS.keys()

CLASSIFIER_ARGS = {
    LogisticRegression: {'max_iter': 300},
    KNeighborsClassifier: {'n_neighbors': 10},
    SVC: {'kernel': 'rbf'},
    DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 10},
    RandomForestClassifier: {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 20},
    LGBMClassifier: {'n_estimators': 20, 'verbose': -1},
    XGBClassifier: {'n_estimators': 20},
    GradientBoostingClassifier: {'n_estimators': 20},
    MLPClassifier: {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 5000},
}
REGRESSOR_ARGS = {
    LinearRegression: {'fit_intercept': True},
    KNeighborsRegressor: {'n_neighbors': 10},
    SVR: {'kernel': 'rbf'},
    DecisionTreeRegressor: {'max_depth': 10},
    RandomForestRegressor: {'max_depth': 10, 'n_estimators': 20},
    LGBMRegressor: {'n_estimators': 20, 'verbose': -1},
    XGBRegressor: {'n_estimators': 20},
    GradientBoostingRegressor: {'n_estimators': 20},
    MLPRegressor: {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 5000},
}

CLASSIFICATION_METRICS = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
}
REGRESSION_METRICS = {
    'neg_mean_absolute_error': lambda y_true, y_pred: -1 * mean_absolute_error(y_true, y_pred),
    'neg_mean_squared_error': lambda y_true, y_pred: -1 * mean_squared_error(y_true, y_pred),
    'neg_root_mean_squared_error': lambda y_true, y_pred: -1 * np.sqrt(mean_squared_error(y_true, y_pred))
}
METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}

FEATURE_SELECTION_METHODS = ['ANOVA', 'RF']

ALL_CELLS = 'All'
OTHER_CELLS = 'Other'
TARGET_COL = 'target'
CELL_TYPE_COL = 'cell_type'
BACKGROUND_COLOR = 'lightgrey'
INTEREST_COLOR = 'red'

LIST_SEP = ', '

# Defaults
NUM_GENES = 5000
REDUCTION = 'umap'
DB = 'ALL'
CLASSIFIER = 'RF'
REGRESSOR = 'RF'
CLASSIFICATION_METRIC = 'f1_weighted'
REGRESSION_METRIC = 'neg_mean_squared_error'
CROSS_VALIDATION = 10
REPEATS = 300
FEATURE_SELECTION = FEATURE_SELECTION_METHODS[0]
SET_FRACTION = 0.5
MIN_SET_SIZE = 10
SEED = 3407
THRESHOLD = 0.05
