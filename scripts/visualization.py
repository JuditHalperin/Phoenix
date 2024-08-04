import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scripts.data import get_column_unique_pathways, get_top_sum_pathways, sum_gene_expression
from scripts.utils import save_plot, get_experiment, get_preprocessed_data, remove_outliers, get_color_mapping, convert2sci, save_csv
from scripts.consts import THRESHOLD, TARGET_COL, CLASSIFICATION_METRICS, ALL_CELLS, OTHER_CELLS, BACKGROUND_COLOR, INTEREST_COLOR, CELL_TYPE_COL, MAP_SIZE


sns.set_theme(style='white')


def plot_p_values(
        heatmap_data: pd.DataFrame,
        cluster_rows: bool = False,
        max_value: int = None,
        title: str = '',
        output: str = None,
    ):  
    if cluster_rows:
        heatmap_data = heatmap_data.loc[np.unique(heatmap_data.index)]

    heatmap_data.index = [i[:50] for i in heatmap_data.index]

    heatmap_data = np.log10(heatmap_data ** (-1))

    if cluster_rows:
        row_linkage = hierarchy.linkage(heatmap_data.fillna(0).replace([np.inf, -np.inf], 0), method='average', metric='euclidean')
        row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']
        heatmap_data = heatmap_data.iloc[row_order, :]

    plt.figure(figsize=(9, 6), dpi=200)
    max_value = max(remove_outliers(heatmap_data.fillna(0).values.flatten().tolist())) if not max_value else max_value
    heatmap = sns.heatmap(heatmap_data, cmap='Reds', cbar=False, vmin=0, vmax=max_value, xticklabels=True, yticklabels=False)  # annot=True, fmt='.2f'

    plt.colorbar(heatmap.collections[0], label='-log10(p-value)')
    if heatmap_data.shape[0] <= MAP_SIZE:
        plt.yticks(np.arange(len(heatmap_data.index)) + 0.5, heatmap_data.index, rotation=0, fontsize=5, ha='right')
        heatmap.set_yticklabels(heatmap_data.index, rotation=0, fontsize=5, ha='right')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='right')
    
    plt.title(title)
    plt.xlabel('')
    save_plot(f'p_values_{title}', output)


def _plot_prediction_scores(
        experiment: dict[str, str | float | list[str]],
        set_name: str,
        by_freq: bool = True,
        rm_outliers: bool = False,
        title: str = '',
    ):
    """
    by_freq: either plot frequency or density
    """
    
    # Draw line for pathway of interest's score
    plt.axvline(
        x=experiment['pathway_score'],
        color=INTEREST_COLOR,
        label=f'{set_name[:50]}: {np.round(experiment["pathway_score"], 3)}, p={convert2sci(experiment["fdr"])}',
        linestyle='--'
    )

    # Draw distribution for background score
    background_scores = experiment['background_scores']
    if rm_outliers:
        background_scores = remove_outliers(background_scores)

    plot_args = {
        'x': background_scores,
        'label': f'Random genes: {np.round(experiment["background_score_mean"], 3)}',
        'color': BACKGROUND_COLOR
    }
    
    if by_freq:
        plt.hist(bins=50 if len(np.unique(background_scores)) > 50 else None, **plot_args)
        plt.ylabel('Frequency')
    else:
        sns.kdeplot(fill=True, **plot_args)
        plt.ylabel('Density')

    metric = experiment['metric']
    plt.xlabel(metric)
    # if metric in CLASSIFICATION_METRICS.keys():
    #     plt.xlim(0, 1)

    plt.legend()
    plt.title(title)


def _plot_expression_across_cell_types(
        expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        cell_type: str,
        title: str = '',
    ):
    """
    pass copy
    """
    if cell_type != ALL_CELLS:
        cell_types.loc[cell_types[CELL_TYPE_COL] != cell_type, CELL_TYPE_COL] = OTHER_CELLS
    
    expression[CELL_TYPE_COL] = cell_types[CELL_TYPE_COL]
    data_long = expression.melt(id_vars=[CELL_TYPE_COL], var_name='genes', value_name='expression')

    color_mapping = get_color_mapping(cell_types[CELL_TYPE_COL].unique().tolist()) if cell_type == ALL_CELLS else {cell_type: INTEREST_COLOR, OTHER_CELLS: BACKGROUND_COLOR}

    sns.violinplot(data=data_long, x=CELL_TYPE_COL, y='expression', hue=CELL_TYPE_COL, palette=color_mapping, width=0.8)
    sns.stripplot(data=data_long, x=CELL_TYPE_COL, y='expression', hue=CELL_TYPE_COL, palette='dark:black', alpha=0.2, size=1, jitter=0.08, zorder=1)  # linewidth=0.5

    plt.ylabel('Expression')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.ylim(bottom=0)
    plt.title(title)


def _plot_expression_across_pseudotime(
        expression: pd.DataFrame,
        pseudotime: pd.DataFrame,
        lineage: str,
        bins: int = 30,
        title: str = '',
    ):
    """
    pass copy
    """
    pseudotime = pseudotime[~pseudotime[lineage].isna()]
    expression = expression.loc[pseudotime.index]

    expression['pseudotime_bin'] = pd.cut(pseudotime[lineage], bins=min(bins, len(pseudotime)), labels=False)
    expression = expression.groupby('pseudotime_bin').mean()

    data_long = expression.reset_index().melt(id_vars='pseudotime_bin', var_name='gene', value_name='expression')

    palette = sns.color_palette('plasma', as_cmap=True)

    sns.boxplot(data=data_long, x='pseudotime_bin', y='expression', hue='pseudotime_bin', palette=palette, width=0.6, legend=None)

    plt.xticks([])
    plt.xlabel('Pseudotime bins')
    plt.ylabel('Expression')
    plt.title(title)


def _plot_expression_distribution(
        expression: pd.DataFrame, 
        target_data: pd.DataFrame,
        target: str,
        target_type: str,
        top_genes: list[str] = [],
    ):
    """
    target_data: `cell_types` or `pseudotime`
    target: `cell_type` or `lineage`
    target_type: `cell_types` or `pseudotime`
    """
    assert target_type in ['cell_types', 'pseudotime']
    globals()[f'_plot_expression_across_{target_type}'](expression[[gene for gene in top_genes if gene in expression.columns]].copy(), target_data.copy(), target)


def _plot_pseudotime(
        reduction: pd.DataFrame,
        pseudotime: pd.DataFrame,
        trajectory: str = None,
        title: bool = False
    ):
    plt.scatter(reduction.iloc[:, 0], reduction.iloc[:, 1], s=10, c=BACKGROUND_COLOR)
    trajectories = [trajectory] if trajectory else pseudotime.columns.tolist()
    for lineage in trajectories:
        plt.scatter(reduction.iloc[:, 0], reduction.iloc[:, 1], s=10, c=pseudotime[lineage], cmap=plt.cm.plasma)
    if title: plt.title(f'{trajectory} Trajectory' if trajectory else 'Trajectories')
    plt.xlabel(reduction.columns[0])
    plt.ylabel(reduction.columns[1])
    plt.colorbar(label='Pseudotime')


def _plot_cell_types(
        reduction: pd.DataFrame,
        cell_types: pd.DataFrame,
        cell_type: str = ALL_CELLS,
        title: bool = False
    ):
    if cell_type != ALL_CELLS:
        cell_types.loc[cell_types[CELL_TYPE_COL] != cell_type, CELL_TYPE_COL] = OTHER_CELLS
    color_mapping = get_color_mapping(cell_types[CELL_TYPE_COL].unique().tolist()) if cell_type == ALL_CELLS else {cell_type: INTEREST_COLOR, OTHER_CELLS: BACKGROUND_COLOR}
    sns.scatterplot(data=reduction, x=reduction.columns[0], y=reduction.columns[1], hue=cell_types[CELL_TYPE_COL], palette=color_mapping, s=15, edgecolor='none')
    plt.legend(title='')
    if title: plt.title(cell_type if cell_type != ALL_CELLS else 'Cell-types')


def _plot_target_data(
        reduction: pd.DataFrame,
        target_data: pd.DataFrame,
        target: str,
        target_type: str,
    ):
    assert target_type in ['cell_types', 'pseudotime']
    globals()[f'_plot_{target_type}'](reduction, target_data.copy(), target)


def _plot_gene_set_expression(
        expression: pd.DataFrame, 
        reduction: pd.DataFrame,
        gene_set: list[str],
        set_name: str = '',
        cells: list[str] = None,
        rm_outliers: bool = True,
    ):
    cells = cells if cells is not None else expression.index
    gene_expression = sum_gene_expression(expression.loc[cells, gene_set])
    clean_expression = remove_outliers(gene_expression) if rm_outliers else gene_expression.tolist()
    
    plt.scatter(reduction.iloc[:, 0], reduction.iloc[:, 1], s=10, c=BACKGROUND_COLOR)
    plt.scatter(reduction.loc[cells].iloc[:, 0], reduction.loc[cells].iloc[:, 1], s=10, c=gene_expression, cmap=plt.cm.Blues, vmin=min(clean_expression), vmax=max(clean_expression))
    
    plt.colorbar(label='Pathway expression sum')
    plt.xlabel(reduction.columns[0])
    plt.ylabel(reduction.columns[1])
    plt.title(set_name)


def plot_experiment(
        output: str,
        target: str,
        set_name: str,
        target_type: str,
        results: pd.DataFrame | str,
        target_data: pd.DataFrame | str,
        expression: pd.DataFrame | str = 'expression',
        reduction: pd.DataFrame | str = 'reduction',
    ):
    """
    target_data: `cell_types` or `pseudotime`
    target: `cell_type` or `lineage`
    target_type: `cell_types` or `pseudotime`    
    """
    target_type = target_type.lower().replace('-', '_')
    assert target_type in ['cell_types', 'pseudotime']

    expression = get_preprocessed_data(expression, output)
    reduction = get_preprocessed_data(reduction, output)
    target_data = get_preprocessed_data(target_data, output)
    experiment = get_experiment(results, output, set_name, target)

    if target_data is None:
        raise ValueError(f'Cannot access `{output}/{target_type}.csv`')
    if experiment is None:
        raise ValueError(f'Cannot access `{output}/{results}.csv`')

    plt.figure(figsize=(13, 10), dpi=200)

    # Gene set prediction score
    plt.subplot(2, 2, 1)
    _plot_prediction_scores(experiment, set_name)

    # Gene set expression distribution
    plt.subplot(2, 2, 2)
    _plot_expression_distribution(expression, target_data, target, target_type, experiment['top_genes'])

    # Gene set expression upon reduction
    plt.subplot(2, 2, 3)
    cells = expression.index if target_type == 'cell_types' else target_data[target].dropna().index
    _plot_gene_set_expression(expression, reduction, experiment['top_genes'], cells=cells)
    
    # Target data upon reduction
    plt.subplot(2, 2, 4)
    _plot_target_data(reduction, target_data, target, target_type)
    
    if target_type == 'pseudotime':
        target_name = "'s pseudotime"
    elif target == ALL_CELLS:
        target_name = 'Identities'
    else:
        target_name = "'s identity"
    plt.suptitle(f'Predicting {target} {target_name} using {set_name}')
    save_plot(f'predicting {target} using {set_name}', output)    


def plot_all_cell_types_and_trajectories(
        reduction: pd.DataFrame, 
        cell_types: pd.DataFrame,
        pseudotime: pd.DataFrame,
        output: str = None,
    ):
    num_plots = int(cell_types is not None) + int(pseudotime is not None)
    plt.figure(figsize=(6.5 * num_plots, 5), dpi=200)

    if cell_types is not None:
        plt.subplot(1, num_plots, 1)
        _plot_cell_types(reduction, cell_types, title=True)
    
    if pseudotime is not None:
        plt.subplot(1, num_plots, num_plots)
        _plot_pseudotime(reduction, pseudotime, title=True)
    
    save_plot('all_cell_types_and_trajectories', output)


def plot(
        output: str,
        expression: pd.DataFrame | str = 'expression', 
        reduction: pd.DataFrame | str = 'reduction', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        classification_results: pd.DataFrame | str = 'cell_type_classification',
        regression_results: pd.DataFrame | str = 'pseudotime_regression',
        threshold: float = THRESHOLD,
    ):
    """
    result table or file title
    """
    expression = get_preprocessed_data(expression, output)
    reduction = get_preprocessed_data(reduction, output)
    cell_types = get_preprocessed_data(cell_types, output)
    pseudotime = get_preprocessed_data(pseudotime, output)

    # Plot target data
    plot_all_cell_types_and_trajectories(reduction, cell_types, pseudotime, output)

    # Plot prediction results
    for result_type, target_data, target_type in zip([classification_results, regression_results], [cell_types, pseudotime], ['Cell-types', 'Pseudotime']):

        results = get_experiment(result_type, output)
        if target_data is None or results is None:
            continue

        data = results.pivot(index='set_name', columns=TARGET_COL, values='fdr')
        save_csv(data, f'p_values_{target_type}', output)

        if data.shape[0] <= MAP_SIZE:  # plot all pathways
            pathways = data.index
            for target in data.columns:
                for pathway_name in pathways:
                    plot_experiment(output, target, pathway_name, target_type, results, target_data, expression, reduction)

        else:  # plot interesting pathways
            pathways = []
            size = (MAP_SIZE - 3) // data.shape[1]
            for target in data.columns:
                if target != ALL_CELLS:
                    pathway_names = get_column_unique_pathways(data, target, size, threshold)
                    pathways.extend(pathway_names)
                    for pathway_name in pathway_names:
                        plot_experiment(output, target, pathway_name, target_type, results, target_data, expression, reduction)
            pathways.extend(get_top_sum_pathways(data, ascending=False, size=3))

            plot_p_values(data, cluster_rows=True, title=f'{target_type} Prediction using All Pathways', output=output)
        
        plot_p_values(data.loc[pathways], title=f'{target_type} Prediction', output=output)
        
        del data
        del results
