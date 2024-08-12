import pandas as pd
from scripts.args import get_run_batch_args
from scripts.data import get_cell_types, get_lineages
from scripts.prediction import get_data, train, compare_scores
from scripts.consts import CLASSIFIERS, REGRESSORS, CLASSIFIER_ARGS, REGRESSOR_ARGS
from scripts.utils import define_background, define_set_size
from scripts.output import load_background_scores, save_background_scores, summarise_result, save_csv


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
        feature_selection: str | None,
        cross_validation: int,
        repeats: int,
        seed: int,
        distribution: str,
        cell_types: pd.DataFrame = None,
        pseudotime: pd.DataFrame = None,
        cell_type: str = None,
        lineage: str = None,
        trim_background: bool = True,
        cache: str = None
    ):

    prediction_args = {
        'expression': expression,
        'predictor': predictor,
        'metric': metric,
        'cross_validation': cross_validation,
        'set_size': set_size,
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
        if trim_background:
            background_scores = [s for s in background_scores if s > min(background_scores) and s < max(background_scores)]
        save_background_scores(background_scores, background, cache)

    # Compare scores
    p_value = compare_scores(pathway_score, background_scores, distribution)

    return p_value, pathway_score, background_scores, top_genes


def run_gene_set_batch(
        batch: int | None,
        batch_gene_sets: dict[str, list[str]],
        expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        pseudotime: pd.DataFrame,
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
        output: str,
        cache: str,
    ) -> None:
    """
    output: main output path for a single batch and temp output path for many batches
    batch: number between 1 and `processes`, or None for a single batch
    """
    if batch_gene_sets is None:
        pass

    classification_results, regression_results = [], []

    logger = f'Batch {batch}: ' if batch else ''
    for i, (set_name, gene_set) in enumerate(batch_gene_sets.items()):
        print(f'{logger}Pathway {i + 1}/{len(batch_gene_sets)}: {set_name}')

        set_size = define_set_size(len(gene_set), set_fraction, min_set_size)
        task_args = {
            'expression': expression, 'gene_set': gene_set,
            'set_size': set_size, 'feature_selection': feature_selection,
            'cross_validation': cross_validation, 'repeats': repeats,
            'seed': seed, 'distribution': distribution, 'cache': cache
        }

        # Cell-type classification
        for cell_type in get_cell_types(cell_types):
            p_value, pathway_score, background_scores, top_genes = run_task(
                predictor=classifier, metric=classification_metric,
                cell_types=cell_types, cell_type=cell_type,
                **task_args
            )
            classification_results.append(summarise_result(
                cell_type, set_name, top_genes, set_size, feature_selection, classifier, classification_metric,
                cross_validation, repeats, seed, pathway_score, background_scores, p_value
            ))
        
        # Pseudo-time regression
        for lineage in get_lineages(pseudotime):
            p_value, pathway_score, background_scores, top_genes = run_task(
                predictor=regressor, metric=regression_metric,
                pseudotime=pseudotime, lineage=lineage,
                **task_args
            )

            regression_results.append(summarise_result(
                lineage, set_name, top_genes, set_size, feature_selection, regressor, regression_metric,
                cross_validation, repeats, seed, pathway_score, background_scores, p_value
            ))

    # Save results
    info = f'_batch{batch}' if batch else ''
    save_csv(classification_results, f'cell_type_classification{info}', output, keep_index=False)
    save_csv(regression_results, f'pseudotime_regression{info}', output, keep_index=False)


if __name__ == '__main__':
    args = get_run_batch_args()
    run_gene_set_batch(**vars(args))
