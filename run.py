import threading
import pandas as pd
import numpy as np
from scripts.args import get_run_args
from scripts.data import preprocess_data, intersect_genes, get_cell_types, get_lineages
from scripts.pathways import get_gene_sets
from scripts.prediction import get_data, train, compare_scores
from scripts.consts import CLASSIFIERS, REGRESSORS, CLASSIFIER_ARGS, REGRESSOR_ARGS
from scripts.utils import define_background, define_set_size, define_batch_size, get_gene_set_batches, load_background_scores, save_background_scores, summarise_result, save_csv
from scripts.visualization import plot


def get_prediction_score(
        expression: pd.DataFrame,
        predictor: str,
        metric: str,
        seed: int,
        gene_set: list[str] = None,
        set_size: int = None,
        feature_selection: str = None,
        cross_validation: int = None,
        cell_types: pd.DataFrame = None,
        pseudotime: pd.DataFrame = None,
        cell_type: str = None,
        lineage: int = None,
    ) -> tuple[float, list[str]]:

    X, y, selected_genes = get_data(
        expression=expression,
        features=gene_set,
        cell_types=cell_types,
        pseudotime=pseudotime,
        cell_type=cell_type,
        lineage=lineage,
        set_size=set_size,
        feature_selection=feature_selection,
        seed=seed,
    )
    
    is_regression = pseudotime is not None
    predictor = REGRESSORS[predictor] if is_regression else CLASSIFIERS[predictor]
    predictor_args = REGRESSOR_ARGS[predictor] if is_regression else CLASSIFIER_ARGS[predictor]
    
    score = train(
        X=X,
        y=y,
        predictor=predictor,
        predictor_args=predictor_args,
        metric=metric,
        cross_validation=cross_validation,
        seed=seed
    )

    return score, selected_genes


def run_task(
        expression: pd.DataFrame,
        gene_set: list[str],
        predictor: str,
        metric: str,
        set_size: int,
        feature_selection: str,
        cross_validation: int,
        repeats: int,
        seed: int,
        cell_types: pd.DataFrame = None,
        pseudotime: pd.DataFrame = None,
        cell_type: str = None,
        lineage: str = None,
        cache: str = None
    ):

    prediction_args = {
        'expression': expression,
        'predictor': predictor,
        'metric': metric,
        'set_size': set_size,
        'cross_validation': cross_validation,
        'cell_types': cell_types,
        'pseudotime': pseudotime,
        'cell_type': cell_type,
        'lineage': lineage,
    }

    # Pathway of interest
    pathway_score, top_genes = get_prediction_score(seed=seed, gene_set=gene_set, feature_selection=feature_selection, **prediction_args)

    # Background
    background = define_background(set_size, repeats, cell_type, lineage)
    background_scores = load_background_scores(background, cache)
    if not background_scores:
        for i in range(repeats):
            background_scores.append(get_prediction_score(seed=i, **prediction_args)[0])
        save_background_scores(background_scores, background, cache)

    # Compare scores
    p_value = compare_scores(pathway_score, background_scores)

    return p_value, pathway_score, background_scores, top_genes


def run_gene_set_batch(
        gene_sets: list[str],
        expression: str,
        cell_types: str,
        pseudotime: str,
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
        cache: str,
        batch: int = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:

    classification_results, regression_results = [], []

    logger = f'Batch {batch}: ' if batch else ''
    for i, (set_name, original_gene_set) in enumerate(gene_sets.items()):

        gene_set = intersect_genes(original_gene_set, expression.columns)
        if not gene_set:
            print(f'{logger}Pathway {i + 1}/{len(gene_sets)}: {set_name} - skipping...')
            continue

        print(f'{logger}Pathway {i + 1}/{len(gene_sets)}: {set_name}')

        set_size = define_set_size(len(gene_set), set_fraction, min_set_size)
        task_args = {
            'expression': expression, 'gene_set': gene_set,
            'set_size': set_size, 'feature_selection': feature_selection,
            'cross_validation': cross_validation, 'repeats': repeats,
            'seed': seed, 'cache': cache
        }

        # Cell-type classification
        for cell_type in get_cell_types(cell_types):
            p_value, pathway_score, background_scores, top_genes = run_task(
                predictor=classifier, metric=classification_metric,
                cell_types=cell_types, cell_type=cell_type,
                **task_args
            )
            classification_results.append(summarise_result(
                cell_type, set_name, original_gene_set, gene_set, top_genes,
                set_size, feature_selection, classifier, classification_metric,
                cross_validation, repeats, seed, pathway_score, background_scores, p_value
            ))
        
        # Pseudo-time regression
        for lineage in get_lineages(pseudotime):
            p_value, pathway_score, background_scores, top_genes = run_task(
                predictor=regressor, metric=regression_metric,
                pseudotime=pseudotime, lineage=lineage
                **task_args
            )
            regression_results.append(summarise_result(
                lineage, set_name, original_gene_set, gene_set, top_genes,
                set_size, feature_selection, regressor, regression_metric,
                cross_validation, repeats, seed, pathway_score, background_scores, p_value
            ))

    return classification_results, regression_results


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
        threads: int,
        output: str,
        cache: str,
    ) -> None:

    expression, cell_types, pseudotime, reduction = preprocess_data(
        expression, cell_types, pseudotime, reduction,
        preprocessed, exclude_cell_types, exclude_lineages, output
    )
    gene_sets = get_gene_sets(pathway_database, custom_pathways, organism)

    batch_args = {
        'expression': expression, 'cell_types': cell_types, 'pseudotime': pseudotime,
        'feature_selection': feature_selection, 'set_fraction': set_fraction, 'min_set_size': min_set_size,
        'classifier': classifier, 'regressor': regressor, 'classification_metric': classification_metric, 'regression_metric': regression_metric,
        'cross_validation': cross_validation, 'repeats': repeats, 'seed': seed, 'cache': cache,
    }

    print('Running experiments...')
    if not threads:
        classification_results, regression_results = run_gene_set_batch(gene_sets=gene_sets, **batch_args)

    else:
        results = []
        thread_objects = []

        try:
            batch_size = define_batch_size(len(gene_sets), threads)
            for batch, batch_gene_sets in enumerate(get_gene_set_batches(gene_sets, batch_size)):
                thread = threading.Thread(target=lambda batch=batch, gene_sets=batch_gene_sets, batch_args=batch_args:
                                          results.append(run_gene_set_batch(batch=batch + 1, gene_sets=gene_sets, **batch_args)))
                thread.start()
                thread_objects.append(thread)

            for thread in thread_objects:
                thread.join()

        except Exception as e:
            print(e)
            for thread in thread_objects:
                thread.join(timeout=1)
                if thread.is_alive():
                    thread.stop()

        classification_results, regression_results = [], []
        for batch_result in results:
            classification_results.extend(batch_result[0])
            regression_results.extend(batch_result[1])

    # Results
    print('Saving results...')
    save_csv(classification_results, 'cell_type_classification', output, keep_index=False)
    save_csv(regression_results, 'pseudotime_regression', output, keep_index=False)

    print('Plotting results...')
    plot(output, expression, reduction, cell_types, pseudotime, classification_results, regression_results)


if __name__ == '__main__':
    args = get_run_args()
    run_tool(**vars(args))
