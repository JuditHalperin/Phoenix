import argparse, os
import pandas as pd
from scripts.consts import *
from scripts.utils import read_csv, get_full_path, get_preprocessed_data, read_gene_sets, get_gene_set_batch, define_batch_size


### Run ###


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
    parser.add_argument('--pathway_database', type=str, nargs='*',
                        help='db name')
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
    parser.add_argument('--processes', type=int, default=0,
                        help='')
    parser.add_argument('--output', type=str, required=True,
                        help='')
    
    return parser.parse_args()


def process_run_args(args):
    args.expression = read_csv(args.expression)
    args.cell_types = read_csv(args.cell_types).loc[args.expression.index] if args.cell_types else None
    if args.cell_types is not None:
        args.cell_types.rename(columns={args.cell_types.columns[0]: CELL_TYPE_COL}, inplace=True)
    args.pseudotime = read_csv(args.pseudotime).loc[args.expression.index] if args.pseudotime else None
    
    if os.path.exists(args.reduction):
        args.reduction = read_csv(args.reduction).loc[args.expression.index]
    else:
        args.reduction = args.reduction.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    args.organism = args.organism.lower()
    if args.pathway_database is not None:
        args.pathway_database = [db.lower() for db in args.pathway_database]
        from scripts.pathways import get_msigdb_organism
        if 'msigdb' in args.pathway_database and len(args.pathway_database) > 1 and get_msigdb_organism(args.organism):
            args.pathway_database = ['msigdb']
            print(f'MSigDB already includes the other annotation databases for {args.organism} - automatically removing other annotations')
    else:
        args.pathway_database = []
    args.custom_pathways = args.custom_pathways if args.custom_pathways else []
    
    args.classifier = args.classifier.upper()
    args.regressor = args.regressor.upper()
    args.classification_metric = args.classification_metric.lower().replace(' ', '_')
    args.regression_metric = args.regression_metric.lower().replace(' ', '_')
    args.feature_selection = args.feature_selection.upper() if args.feature_selection else None

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    args.output = get_full_path(args.output)

    args.cache = os.path.join(args.output, 'cache')
    if not os.path.exists(args.cache):
        os.mkdir(args.cache)

    args.tmp = os.path.join(args.output, 'tmp')
    if args.processes and not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    return args


def validate_run_args(args):
    assert args.cell_types is not None or args.pseudotime is not None, 'Provide at least `cell_types` or `pseudotime`'
    assert args.cell_types is None or ALL_CELLS not in args.cell_types[CELL_TYPE_COL].tolist(), f'`cell_types` cannot contain a cell-type called `{ALL_CELLS}`'
    assert isinstance(args.reduction, pd.DataFrame) or args.reduction in REDUCTION_METHODS
    assert args.pathway_database is not None or args.custom_pathways is not None, 'Provide at least `pathway_database` or `custom_pathways`'
    assert args.pathway_database is None or all([db in DATABASES for db in args.pathway_database])
    assert args.classifier in CLASSIFIERS
    assert args.regressor in REGRESSORS
    assert args.classification_metric in CLASSIFICATION_METRICS.keys()
    assert args.regression_metric in REGRESSION_METRICS.keys()
    assert not args.feature_selection or args.feature_selection in FEATURE_SELECTION_METHODS
    assert 1 <= args.cross_validation <= 10
    assert args.repeats > 1
    assert args.seed > 0
    assert 0 < args.set_fraction <= 1


def get_run_args():
    args = parse_run_args()
    args = process_run_args(args)
    validate_run_args(args)
    return args


### Batch run ###


def parse_run_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=None, help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')

    parser.add_argument('--feature_selection', type=str, required=True, help='')
    parser.add_argument('--set_fraction', type=float, required=True, help='')
    parser.add_argument('--min_set_size', type=int, required=True, help='')

    parser.add_argument('--classifier', type=str, required=True, help='')
    parser.add_argument('--regressor', type=str, required=True, help='')
    parser.add_argument('--classification_metric', type=str, required=True, help='')
    parser.add_argument('--regression_metric', type=str, required=True, help='')
    parser.add_argument('--cross_validation', type=int, required=True, help='')
    parser.add_argument('--repeats', type=int, required=True, help='')
    parser.add_argument('--seed', type=int, required=True, help='')

    parser.add_argument('--output', type=str, required=True, help='')
    parser.add_argument('--cache', type=str, required=True, help='')
    parser.add_argument('--tmp', type=str, required=True, help='')

    return parser.parse_args()


def process_run_batch_args(args):
    args.expression = get_preprocessed_data('expression', args.output)
    args.cell_types = get_preprocessed_data('cell_types', args.output)
    args.pseudotime = get_preprocessed_data('pseudotime', args.output)

    gene_sets = read_gene_sets(os.path.join(args.output, 'gene_sets.csv'))
    args.batch_gene_sets = get_gene_set_batch(gene_sets, args.batch, args.batch_size)
    del args.batch_size

    if args.batch is not None:
        args.output = args.tmp
        del args.tmp

    return args


def get_run_batch_args():
    args = parse_run_batch_args()
    args = process_run_batch_args(args)
    return args


### Plot ###


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
    args.cell_type = args.cell_type if args.cell_type else []
    args.lineage = args.lineage if args.lineage else []
    args.pathway = args.pathway if args.pathway else []
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
