import random, inspect, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
import scipy.stats as stats
from scripts.consts import ALL_CELLS, SEED, CELL_TYPE_COL, METRICS
from scripts.utils import show_runtime


@show_runtime
def get_target(
        cell_types: pd.DataFrame = None,
        pseudotime: pd.DataFrame = None,
        cell_type: str = None,
        lineage: int = None,
        scale: bool = True
    ):
    """
    scale: whether to normalize continuous target pseudo-time values using min-max scaler
    """
    if pseudotime is not None:
        y = pseudotime.loc[:, lineage].dropna()
        return pd.Series(MinMaxScaler().fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index) if scale else y
    
    if cell_type == ALL_CELLS:
        return cell_types[CELL_TYPE_COL]
    return cell_types[CELL_TYPE_COL] == cell_type


@show_runtime
def get_data(
        expression: pd.DataFrame,
        features: list[str] = None,
        cell_types: pd.DataFrame = None,
        pseudotime: pd.DataFrame = None,
        cell_type: str = None,
        lineage: int = None,
        scale_features: bool = True,
        scale_target: bool = True,
        set_size: int = None,
        feature_selection: str = None,
        selection_args: dict = {},
        ordered_selection: bool = False,
        seed: int = SEED
    ) -> tuple:
    """
    feature_selection: either 'ANOVA' or 'RF', supported for both classification and regression
    ordered_selection: ignored if feature_selection is set
    """
    assert (cell_types is not None and cell_type is not None) or (pseudotime is not None and lineage is not None)
    is_regression = pseudotime is not None

    y = get_target(cell_types, pseudotime, cell_type, lineage, scale_target)
    cells = y.index
    features = [f for f in features if f in expression.columns] if features is not None else expression.columns
    X = expression.loc[cells, features]
    X = StandardScaler().fit_transform(X) if scale_features else X

    # Select all features
    if not set_size or set_size >= len(features):
        return X, y, features

    # Select best features using either ANOVA or RF
    if feature_selection:
        if feature_selection == 'ANOVA':
            selected_features = SelectKBest(score_func=f_regression if is_regression else f_classif, k=set_size).fit(X, y)
            selected_genes = [features[i] for i in selected_features.get_support(indices=True)]
            return selected_features.transform(X), y, selected_genes
        
        if feature_selection == 'RF':
            if 'n_estimators' not in selection_args.keys():
                selection_args['n_estimators'] = 50
            if is_regression:
                importances = RandomForestRegressor(random_state=seed, **selection_args).fit(X, y).feature_importances_
            else:
                importances = RandomForestClassifier(random_state=seed, class_weight='balanced', **selection_args).fit(X, y).feature_importances_
            selected_indices = (-importances).argsort()[:set_size]
            selected_genes = [features[i] for i in selected_indices]
            return X[:, selected_indices], y, selected_genes
        
        raise ValueError(f'Unsupported feature selection method {feature_selection}')

    # Select first
    if ordered_selection:
        return X[:, :set_size], y, features[:set_size]

    # Select randomly
    return X[:, random.Random(seed).sample(list(range(X.shape[1])), set_size)], y, None


@show_runtime
def train(
        X, y,
        predictor,
        predictor_args: dict,
        metric: str,
        cross_validation: int = None,
        balanced_weights: bool = True,
        train_size: float = 0.8,
        bins: int = 3,
        seed: int = SEED,
    ) -> float:

    if 'n_jobs' in inspect.signature(predictor).parameters:
        predictor_args['n_jobs'] = -1  # all processes
    if 'random_state' in inspect.signature(predictor).parameters:
        predictor_args['random_state'] = seed
    if balanced_weights and 'class_weight' in inspect.signature(predictor).parameters:
        predictor_args['class_weight'] = 'balanced'

    model = predictor(**predictor_args)
    score = make_scorer(METRICS[metric], greater_is_better=True)

    encode_labels = isinstance(y.iloc[0], str)
    if encode_labels:
        le = LabelEncoder()
        y = le.fit_transform(y)

    if cross_validation:
        score = np.median(cross_val_score(model, X, y, cv=cross_validation, scoring=score))

    else:
        stratify = pd.cut(y, bins=bins, labels=False) if y.dtype == float else y
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, train_size=train_size, random_state=seed)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = score(y_test, y_pred)

    return float(score)


def compare_scores(pathway_score: float, background_scores: list[float], distribution: str) -> float:

    if all([s == pathway_score for s in background_scores]):
        p_value = np.NaN

    elif distribution == 'normal':
        alternative = 'less'  # background is less than pathway
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Precision loss occurred in moment calculation')
            p_value = stats.ttest_1samp(background_scores, pathway_score, alternative=alternative)[1]

    elif distribution == 'gamma':
        try:
            shape, loc, scale = stats.gamma.fit(background_scores)
            cdf_value = stats.gamma.cdf(pathway_score, shape, loc, scale)
            p_value = 1 - cdf_value
        except stats._warnings_errors.FitError:
            p_value = np.NaN
        
    else:
        raise ValueError('Unsupported distribution type. Use `normal` or `gamma`')

    return p_value if not np.isnan(p_value) else 1.0
