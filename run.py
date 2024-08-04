import subprocess
from scripts.args import get_run_args
from scripts.data import preprocess_data
from scripts.pathways import get_gene_sets
from scripts.utils import define_batch_size, get_batch_run_cmd, get_aggregation_cmd
from scripts.output import aggregate_result
from scripts.visualization import plot
from scripts.consts import CELL_TYPE_COL


def summarize(output: str, tmp: str):
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
        processes: int,
        output: str,
        cache: str,
        tmp: str,
    ) -> None:

    expression, cell_types, pseudotime, _ = preprocess_data(expression, cell_types, pseudotime, reduction, preprocessed, exclude_cell_types, exclude_lineages, output)
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism, expression.columns, output)
    batch_size = define_batch_size(len(gene_sets), processes)
    print(f'Running experiments for {len(gene_sets)} gene annotations with batch size of {batch_size}...')

    batch_args = {
        'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
        'classifier': classifier, 'regressor': regressor, 'classification_metric': classification_metric, 'regression_metric': regression_metric,
        'cross_validation': cross_validation, 'repeats': repeats, 'seed': seed, 'distribution': distribution,
        'output': output, 'cache': cache, 'tmp': tmp,
    }

    cmd = get_batch_run_cmd(processes, batch_size, task_len=len(cell_types[CELL_TYPE_COL].unique()) + pseudotime.shape[1], **batch_args)
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    job_id = process.stdout.strip().split()[-1]

    cmd = get_aggregation_cmd(output, tmp, job_id, processes)
    subprocess.run(cmd, shell=True) #, capture_output=True, text=True)


if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
