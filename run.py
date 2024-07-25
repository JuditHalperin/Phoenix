import subprocess
from run_batch import run_gene_set_batch
from scripts.args import get_run_args
from scripts.data import preprocess_data
from scripts.pathways import get_gene_sets
from scripts.prediction import  adjust_p_value
from scripts.utils import define_batch_size, get_gene_set_batches, save_csv, get_batch_run_cmd
from scripts.visualization import plot


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
        processes: int,
        output: str,
        cache: str,
        tmp: str,
    ) -> None:

    expression, _, _, _ = preprocess_data(expression, cell_types, pseudotime, reduction, preprocessed, exclude_cell_types, exclude_lineages, output)
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism, expression.columns, output)
    batch_size = define_batch_size(len(gene_sets), processes)
    print(f'Running experiments for {len(gene_sets)} gene annotations with batch size of {batch_size}...')

    batch_args = {
        'batch_size': batch_size,
        'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
        'classifier': classifier, 'regressor': regressor, 'classification_metric': classification_metric, 'regression_metric': regression_metric,
        'cross_validation': cross_validation, 'repeats': repeats, 'seed': seed,
        'output': output, 'cache': cache, 'tmp': tmp,
    }

    cmd = get_batch_run_cmd(processes, **batch_args)
    subprocess.run(cmd, shell=True)


    # Multiple comparison correction
    # print('Correcting p-values...')
    # classification_results['fdr'] = adjust_p_value(classification_results['p_value'].values)
    # regression_results['fdr'] = adjust_p_value(regression_results['p_value'].values)

    # # Results
    # print('Saving results...')
    # save_csv(classification_results, 'cell_type_classification', output, keep_index=False)
    # save_csv(regression_results, 'pseudotime_regression', output, keep_index=False)

    # print('Plotting results...')
    # plot(output, expression, reduction, cell_types, pseudotime, classification_results, regression_results)


if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
