import pandas as pd
from scripts.args import get_run_args
from scripts.data import preprocess_data
from scripts.pathways import get_gene_sets
from scripts.utils import define_batch_size
from scripts.computation import run_setup_cmd, run_experiments_cmd, run_aggregation_cmd, get_gene_set_batch
from scripts.output import read_raw_data, aggregate_result, get_preprocessed_data, read_gene_sets
from scripts.visualization import plot
from run_batch import run_gene_set_batch


def setup(
        expression: str,
        cell_types: str,
        pseudotime: str,
        reduction: str,
        preprocessed: bool,
        exclude_cell_types: list[str],
        exclude_lineages: list[str],
        pathway_database: list[str],
        custom_pathways: list[str],
        organism: str,
        min_set_size: int,
        seed: int,
        output: str,
        return_data: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]] | None:

    expression, cell_types, pseudotime, reduction = read_raw_data(expression, cell_types, pseudotime, reduction)
    expression, cell_types, pseudotime, reduction = preprocess_data(expression, cell_types, pseudotime, reduction, preprocessed=preprocessed, exclude_cell_types=exclude_cell_types, exclude_lineages=exclude_lineages, seed=seed, output=output)
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism, expression.columns, min_set_size, output)
    
    if return_data:
        return expression, cell_types, pseudotime, reduction, gene_sets
    return None


def run_experiments(
        batch: int | None,
        feature_selection: str,
        set_fraction: float,
        min_set_size: int,
        classifier: str,
        regressor: str,
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        repeats: int,
        seed: int,
        distribution: str,
        processes: int,
        output: str,
        tmp: str,
        cache: str,
        expression: pd.DataFrame | str = 'expression', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        gene_sets: dict[str, list[str]] | str = 'gene_sets',
    ) -> None:
    """
    Run experiments for a single batch of gene sets.
    batch: index between 1 and `processes`, or None for a single batch
    output: main output path
    """
    expression = get_preprocessed_data(expression, output)
    cell_types = get_preprocessed_data(cell_types, output)
    pseudotime = get_preprocessed_data(pseudotime, output)

    gene_sets = read_gene_sets(gene_sets)
    batch_size = define_batch_size(len(gene_sets), processes)
    batch_gene_sets = get_gene_set_batch(gene_sets, batch, batch_size)

    if not batch or batch == 1:
        print(f'Running experiments for {len(gene_sets)} gene annotations with batch size of {batch_size}...')

    run_gene_set_batch(
        batch, batch_gene_sets, expression, cell_types, pseudotime,
        feature_selection, set_fraction, min_set_size,
        classifier, regressor, classification_metric, regression_metric,
        cross_validation, repeats, seed, distribution,
        tmp if batch else output, cache,
    )


def summarize(
        output: str,
        tmp: str = None,
    ) -> None:
    aggregate_result('cell_type_classification', output, tmp)
    aggregate_result('pseudotime_regression', output, tmp)
    plot(output)


def run_tool(
        expression: str,
        cell_types: str,
        pseudotime: str,
        reduction: str,
        preprocessed: bool,
        exclude_cell_types: list[str],
        exclude_lineages: list[str],
        organism: str,
        pathway_database: list[str],
        custom_pathways: list[str],
        feature_selection: str,
        set_fraction: float,
        min_set_size: int,
        classifier: str,
        regressor: str,
        classification_metric: str,
        regression_metric: str,
        cross_validation: int,
        repeats: int,
        seed: int,
        distribution: str,
        sbatch: bool,
        processes: int,
        output: str,
        cache: str,
        tmp: str,
    ) -> None:

    if sbatch:
        # Setup
        setup_args = {
            'expression': expression, 'cell_types': cell_types, 'pseudotime': pseudotime, 'reduction': reduction,
            'preprocessed': preprocessed, 'exclude_cell_types': exclude_cell_types, 'exclude_lineages': exclude_lineages,
            'pathway_database': pathway_database, 'custom_pathways': custom_pathways, 'organism': organism, 'min_set_size': min_set_size,
            'seed': seed, 'output': output
        }
        setup_job_id = run_setup_cmd(setup_args, tmp)

        # Experiments
        exp_args = {
            'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
            'classifier': classifier, 'regressor': regressor, 'classification_metric': classification_metric, 'regression_metric': regression_metric,
            'cross_validation': cross_validation, 'repeats': repeats, 'seed': seed, 'distribution': distribution,
            'processes': processes, 'output': output, 'tmp': tmp, 'cache': cache,
        }
        exp_job_id = run_experiments_cmd(setup_job_id, exp_args, tmp)

        # Aggregation
        run_aggregation_cmd(exp_job_id, processes, output, tmp)
    
    else:
        expression, cell_types, pseudotime, reduction, gene_sets = setup(expression, cell_types, pseudotime, reduction, preprocessed, exclude_cell_types, exclude_lineages, pathway_database, custom_pathways, organism, min_set_size, seed, output, return_data=True)
        run_experiments(None, feature_selection, set_fraction, min_set_size, classifier, regressor, classification_metric, regression_metric, cross_validation, repeats, seed, distribution, processes, output, tmp, cache, expression, cell_types, pseudotime, gene_sets)
        summarize(output)


if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
