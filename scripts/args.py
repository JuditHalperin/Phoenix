import argparse, os
import pandas as pd
from scripts.consts import *
from scripts.utils import read_csv, get_full_path


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--expression', type=str, required=True,
                        help='Path to single-cell expression data in (CSV file) where genes in columns and cells in rows')
    parser.add_argument('--cell_types', type=str,
                        help='')
    parser.add_argument('--pseudotime', type=str,
                        help='')
    parser.add_argument('--reduction', type=str, default=REDUCTION,
                        help='Path to dimensionality reduction data or reduction method name')

    # Data preprocessing
    parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='Log-normalized expression data')
    parser.add_argument('--exclude_cell_types', type=str, nargs='*',
                        help='Cell type to exclude from analysis')
    parser.add_argument('--exclude_lineages', type=str, nargs='*',
                        help='Lineage to exclude from analysis')

    # Pathway annotations
    parser.add_argument('--organism', type=str, required=True,
                        help='')
    parser.add_argument('--pathway_database', type=str, default=None,
                        help='db name or all')
    parser.add_argument('--custom_pathways', type=str, nargs='*',
                        help='path or ids')

    # Feature selection
    parser.add_argument('--feature_selection', type=str, default=FEATURE_SELECTION,
                        help='')
    parser.add_argument('--set_fraction', type=float, default=SET_FRACTION,
                        help='')
    parser.add_argument('--min_set_size', type=int, default=MIN_SET_SIZE,
                        help='')

    # Prediction model
    parser.add_argument('--classifier', type=str, default=CLASSIFIER,
                        help='')
    parser.add_argument('--regressor', type=str, default=REGRESSOR,
                        help='')
    parser.add_argument('--classification_metric', type=str, default=CLASSIFICATION_METRIC,
                        help='')
    parser.add_argument('--regression_metric', type=str, default=REGRESSION_METRIC,
                        help='')
    parser.add_argument('--cross_validation', type=int, default=CROSS_VALIDATION,
                        help='')
    parser.add_argument('--repeats', type=int, default=REPEATS,
                        help='')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='')
    
    # Output
    parser.add_argument('--threads', type=int, default=None,
                        help='')
    parser.add_argument('--output', type=str, required=True,
                        help='')
    
    return parser.parse_args()


def process_run_args(args):
    args.expression = read_csv(args.expression)
    args.cell_types = read_csv(args.cell_types).loc[args.expression.index].rename(columns=[CELL_TYPE_COL]) if args.cell_types else None
    args.pseudotime = read_csv(args.pseudotime).loc[args.expression.index] if args.pseudotime else None
    try:
        args.reduction = read_csv(args.reduction).loc[args.expression.index]
    except:
        args.reduction = args.reduction.lower().replace('-', '').replace('_', '').replace(' ', '')
    if args.pathway_database:
        args.pathway_database = args.pathway_database.lower()
        args.pathway_database = DATABASES if args.pathway_database == ALL_DATABASES else [args.pathway_database]
    else:
        args.pathway_database = []
    args.custom_pathways = args.custom_pathways if args.custom_pathways else []
    args.organism = args.organism.lower()
    args.classifier = args.classifier.upper()
    args.regressor = args.regressor.upper()
    args.classification_metric = args.classification_metric.lower().replace(' ', '_')
    args.regression_metric = args.regression_metric.lower().replace(' ', '_')
    args.feature_selection = args.feature_selection.upper()
    args.output = get_full_path(args.output)
    args.cache = os.path.join(args.output, 'cache')
    return args


def validate_run_args(args):
    assert args.cell_types or args.pseudotime, 'Provide at least `cell_types` or `pseudotime`'
    assert ALL_CELLS not in args.cell_types[CELL_TYPE_COL].tolist(), f'`cell_types` cannot contain a cell-type called `{ALL_CELLS}`'
    assert isinstance(args.reduction, pd.DataFrame) or args.reduction in REDUCTION_METHODS
    assert args.pathway_database or args.custom_pathways, 'Provide at least `pathway_database` or `custom_pathways`'
    assert not args.pathway_database or all([db in DATABASES for db in args.pathway_database])
    assert args.classifier in CLASSIFIERS
    assert args.regressor in REGRESSORS
    assert args.classification_metric in CLASSIFICATION_METRICS.keys()
    assert args.regression_metric in REGRESSION_METRICS.keys()
    assert args.feature_selection in FEATURE_SELECTION_METHODS
    assert 1 <= args.cross_validation <= 10
    assert args.repeats > 1
    assert args.seed > 0
    assert 0 < args.set_fraction <= 1
    assert not args.threads or args.threads >= 1


def get_run_args():
    args = parse_run_args()
    args = process_run_args(args)
    validate_run_args(args)
    return args


def parse_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--cell_type', type=str, nargs='*',
                        help='')
    parser.add_argument('--lineage', type=str, nargs='*',
                        help='')
    parser.add_argument('--pathway', type=str, nargs='*',
                        help='')
    parser.add_argument('--all_plots', action='store_true', default=False,
                        help='')
    parser.add_argument('--output', type=str, required=True,
                        help='')
    
    return parser.parse_args()


def process_plot_args(args):
    args.output = get_full_path(args.output)
    return args


def validate_plot_args(args):
    if not args.all_plots:
        assert args.pathway, 'Provide `pathway`'
        assert args.cell_type or args.lineage, 'Provide either `cell_type` or `lineage` target column'


def get_plot_args():
    args = parse_plot_args()
    args = process_plot_args(args)
    validate_plot_args(args)
    return args
