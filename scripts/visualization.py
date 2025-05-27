import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.cluster import hierarchy
from scripts.data import sum_gene_expression
from scripts.utils import remove_outliers, get_color_mapping, convert_to_sci
from scripts.output import save_plot, get_experiment, get_preprocessed_data, save_csv 
from scripts.consts import THRESHOLD, TARGET_COL, ALL_CELLS, OTHER_CELLS, BACKGROUND_COLOR, INTEREST_COLOR, CELL_TYPE_COL, MAP_SIZE, DPI, LEGEND_FONT_SIZE, POINT_SIZE


sns.set_theme(style='white')
sys.setrecursionlimit(10000)


def get_top_sum_pathways(data, ascending: bool, size: int) -> list[str]:
    return data.copy().dropna(axis=0).sum(axis=1).sort_values(ascending=ascending).head(size).index.tolist()


def get_column_unique_pathways(data, col: str, size: int, threshold: float | None) -> list[str]:
    """Get pathways that are unique to the current cell type compared to the rest"""
    tmp = data.copy()

    # Keep experiments with most significant results at current cell type compared to the rest and below a certain threshold
    tmp = tmp[(tmp[col] == tmp.min(axis=1)) & (tmp[col] <= threshold if threshold else 1)]

    # Keep 10% top experiments to focus on the most significant results
    most_sig = int(data.shape[0] * 0.1) if data.shape[0] > 100 else data.shape[0]
    tmp = tmp.loc[tmp[col].sort_values(ascending=True).index[:most_sig]]

    # Keep experiments with the highest difference between the minimum and the current cell type
    to_drop = [col, ALL_CELLS] if ALL_CELLS in tmp.columns else [col]
    tmp['max_diff'] = tmp.drop(to_drop, axis=1).min(axis=1) - tmp[col]
    tmp = tmp.sort_values(by='max_diff', ascending=False)
    return tmp.head(size).index.tolist()


def get_all_column_unique_pathways(data, size: int, threshold: float):
    return [get_column_unique_pathways(data, col, size // data.shape[1], threshold)
            for col in data.columns if col != ALL_CELLS]


def plot_p_values(
        heatmap_data: pd.DataFrame,
        cluster_rows: bool = False,
        max_value: int = None,
        target_fontsize: int = 10,
        title: str = '',
        output: str = None,
        format: str = 'png',
    ):
    
    if cluster_rows:
        heatmap_data = heatmap_data.loc[np.unique(heatmap_data.index)]
    heatmap_data.index = [i[:50] for i in heatmap_data.index]

    if ALL_CELLS in heatmap_data.columns:
        heatmap_data = heatmap_data[[ALL_CELLS] + heatmap_data.drop(ALL_CELLS, axis=1).columns.tolist()]

    heatmap_data.replace(0, heatmap_data[heatmap_data != 0].stack().min() / 10, inplace=True)
    heatmap_data = np.log10(heatmap_data ** (-1))

    if cluster_rows:
        row_linkage = hierarchy.linkage(heatmap_data.fillna(0).replace([np.inf, -np.inf], 0), method='average', metric='euclidean')
        row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']
        heatmap_data = heatmap_data.iloc[row_order, :]

    plt.figure(figsize=(8, 6 if heatmap_data.shape[0] > 1 else 3), dpi=DPI)
    max_value = max(heatmap_data.fillna(0).values.flatten().tolist()) if not max_value else max_value
    heatmap = sns.heatmap(heatmap_data, cmap='Reds', cbar=False, vmin=0, vmax=int(max_value), xticklabels=True, yticklabels=False)

    plt.colorbar(heatmap.collections[0], label='-log10(p-value)')
    if heatmap_data.shape[0] <= MAP_SIZE:
        plt.yticks(np.arange(len(heatmap_data.index)) + 0.5, heatmap_data.index, rotation=0, fontsize=6, ha='right')
        heatmap.set_yticklabels(heatmap_data.index, rotation=0, fontsize=6, ha='right')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=target_fontsize, ha='center')
    
    plt.title(title, fontsize=target_fontsize + 3)
    plt.xlabel('')

    if heatmap_data.shape[0] == 1:
        heatmap.set_yticklabels(heatmap_data.index, rotation=90, fontsize=13, ha='right')
        plt.yticks(np.arange(1), heatmap_data.index, rotation=90, fontsize=13, ha='right')

    save_plot(f'p_values_{title}', output, format=format)


def _plot_prediction_scores(
        experiment: dict[str, str | float | list[str]],
        by_freq: bool = True,
        show_fit: bool = True,
        add_legend: bool = False,
        title: str = '',
    ):
    """
    by_freq: either plot frequency or density
    """
    
    # Draw line for pathway of interest's score
    plt.axvline(
        x=experiment['pathway_score'],
        color=INTEREST_COLOR,
        label=f'Pathway: {np.round(experiment["pathway_score"], 3)}, p={convert_to_sci(experiment["fdr"])}',
        linestyle='--'
    )

    # Draw distribution for background score
    background_scores = experiment['background_scores']
    plot_args = {
        'x': background_scores,
        'label': f'Background: {np.round(experiment["background_score_mean"], 3)}',
        'color': BACKGROUND_COLOR
    }
    
    if by_freq:
        plt.hist(bins=50 if len(np.unique(background_scores)) > 50 else None, **plot_args)
        plt.ylabel('Frequency')
    else:
        sns.kdeplot(fill=True, **plot_args)
        plt.ylabel('Density')

    if show_fit and 'distribution' in experiment.keys() and experiment['distribution'] == 'gamma':
        shape, loc, scale = stats.gamma.fit(background_scores)
        x = np.linspace(min(background_scores), max(background_scores), 1000)
        pdf = stats.gamma.pdf(x, shape, loc=loc, scale=scale)
        bin_edges = np.histogram_bin_edges(background_scores, bins=30)  # get bin edges for consistent plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        fit_values = np.interp(bin_centers, x, pdf * len(background_scores) * np.diff(bin_edges)[0])  # scale fit to match histogram frequency
        plt.plot(bin_centers, fit_values, color='grey', lw=2, label='Gamma fit')

    plt.xlabel(experiment['metric'])
    if add_legend:
        plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.title(title)


def _plot_expression_across_cell_types(
        expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        cell_type: str,
        title: str = '',
        sep_genes: int = 5,
        dots: bool = False,
    ):
    """
    pass copy
    """
    if cell_type != ALL_CELLS:
        cell_types.loc[cell_types[CELL_TYPE_COL] != cell_type, CELL_TYPE_COL] = OTHER_CELLS
    
    expression[CELL_TYPE_COL] = cell_types[CELL_TYPE_COL]
    data_long = expression.melt(id_vars=[CELL_TYPE_COL], var_name='genes', value_name='expression')

    color_mapping = (get_color_mapping(cell_types[CELL_TYPE_COL].unique().tolist()) if cell_type == ALL_CELLS
                     else {cell_type: INTEREST_COLOR, OTHER_CELLS: BACKGROUND_COLOR})

    # if expression.shape[1] - 1 <= sep_genes:
    #     for key in list(color_mapping.keys()):
    #         for gene in data_long['genes'].unique().tolist():
    #             color_mapping[f'{key}_{gene}'] = color_mapping[key]

    #     data_long['combination'] = data_long[CELL_TYPE_COL] + '_' + data_long['genes']

    #     for cell_type in data_long[CELL_TYPE_COL].unique().tolist():
    #         for gene in data_long['genes'].unique().tolist():        
    #             sns.boxenplot(data=data_long[(data_long['genes'] == gene) & (data_long[CELL_TYPE_COL] == cell_type)], x='combination', y='expression', hue=CELL_TYPE_COL, palette=color_mapping, width=0.8)
    # else:
    sns.violinplot(data=data_long, x=CELL_TYPE_COL, y='expression', hue=CELL_TYPE_COL, palette=color_mapping, width=0.8)
    if dots:
        sns.stripplot(data=data_long, x=CELL_TYPE_COL, y='expression', hue=CELL_TYPE_COL, palette='dark:black', alpha=0.2, size=1, jitter=0.08, zorder=1)  # linewidth=0.5

    plt.ylabel('Expression')
    plt.xlabel('')
    plt.xticks(fontsize=LEGEND_FONT_SIZE)
    if cell_type == ALL_CELLS:
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
    cells = expression.index.intersection(pseudotime.index)
    pseudotime = pseudotime.loc[cells]
    pseudotime = pseudotime[~pseudotime[lineage].isna()]
    expression = expression.loc[cells]

    expression['pseudotime_bin'] = pd.cut(pseudotime[lineage], bins=min(bins, len(pseudotime)), labels=False)
    expression = expression.groupby('pseudotime_bin').mean()

    data_long = expression.reset_index().melt(id_vars='pseudotime_bin', var_name='gene', value_name='expression')

    palette = sns.color_palette('plasma', as_cmap=True)

    sns.boxplot(data=data_long, x='pseudotime_bin', y='expression', hue='pseudotime_bin', palette=palette, width=0.6, legend=None, showfliers=False)

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
        title: bool = False,
        subtitle: bool = False,
    ):
    plt.scatter(reduction.iloc[:, 0], reduction.iloc[:, 1], s=POINT_SIZE, c=BACKGROUND_COLOR)
    trajectories = [trajectory] if trajectory else pseudotime.columns.tolist()
    cells = reduction.index.intersection(pseudotime.index)
    for lineage in trajectories:
        plt.scatter(reduction.loc[cells, reduction.columns[0]], reduction.loc[cells, reduction.columns[1]], s=POINT_SIZE, c=pseudotime.loc[cells, lineage], cmap=plt.cm.plasma)
    if title: plt.title(f'{trajectory} Trajectory' if trajectory else 'Trajectories')
    if subtitle: plt.suptitle(f'n = {len(reduction.index):,}', y=0.83, x=0.7, fontsize=11)
    plt.xlabel(reduction.columns[0])
    plt.ylabel(reduction.columns[1])
    plt.colorbar(label='Pseudotime')


def _plot_cell_types(
        reduction: pd.DataFrame,
        cell_types: pd.DataFrame,
        cell_type: str = ALL_CELLS,
        title: bool = False,
        subtitle: bool = False
    ):
    if cell_type != ALL_CELLS:
        cell_types.loc[cell_types[CELL_TYPE_COL] != cell_type, CELL_TYPE_COL] = OTHER_CELLS
    color_mapping = get_color_mapping(cell_types[CELL_TYPE_COL].unique().tolist()) if cell_type == ALL_CELLS else {cell_type: INTEREST_COLOR, OTHER_CELLS: BACKGROUND_COLOR}
    sns.scatterplot(data=reduction, x=reduction.columns[0], y=reduction.columns[1], hue=cell_types[CELL_TYPE_COL], palette=color_mapping, s=POINT_SIZE, edgecolor='none')
    plt.legend(title='', fontsize=LEGEND_FONT_SIZE)
    if title: plt.title(cell_type if cell_type != ALL_CELLS else 'Cell-types')
    if subtitle: plt.suptitle(f'n = {len(cell_types):,}', y=0.83, x=0.7, fontsize=11)


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
    ):
    cells = cells if cells is not None else expression.index
    gene_expression = sum_gene_expression(expression.loc[cells, gene_set])
    if len(gene_set) <= 1 or isinstance(gene_expression, np.float64):  
        return
    clean_expression = remove_outliers(gene_expression)
    
    plt.scatter(reduction.iloc[:, 0], reduction.iloc[:, 1], s=POINT_SIZE, c=BACKGROUND_COLOR)
    plt.scatter(reduction.loc[cells].iloc[:, 0], reduction.loc[cells].iloc[:, 1], s=POINT_SIZE, c=gene_expression, cmap=plt.cm.Blues, vmin=min(clean_expression), vmax=max(clean_expression))
    
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
        as_single_row: bool = False,
        format: str = 'png',
    ):
    """
    target_data: `cell_types` or `pseudotime`
    target: `cell_type` or `lineage`
    target_type: `cell_types` or `pseudotime`
    as_single_row: whether to plot all subplots in a single row
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

    plt.figure(figsize=(8, 6), dpi=DPI) if not as_single_row else plt.figure(figsize=(14, 3), dpi=DPI)

    # Gene set prediction score
    plt.subplot(2, 2, 1) if not as_single_row else plt.subplot(1, 4, 1)
    _plot_prediction_scores(experiment, add_legend=not as_single_row)

    # Gene set expression distribution
    plt.subplot(2, 2, 2) if not as_single_row else plt.subplot(1, 4, 2)
    _plot_expression_distribution(expression, target_data, target, target_type, experiment['top_genes'])

    # Gene set expression upon reduction
    plt.subplot(2, 2, 3) if not as_single_row else plt.subplot(1, 4, 3)
    cells = list(set(reduction.index).intersection(set(expression.index)))
    reduction, expression, target_data = reduction.loc[cells], expression.loc[cells], target_data.loc[cells]
    cells = cells if target_type == 'cell_types' else target_data[target].dropna().index
    _plot_gene_set_expression(expression, reduction, experiment['top_genes'], cells=cells)
    
    # Target data upon reduction
    plt.subplot(2, 2, 4) if not as_single_row else plt.subplot(1, 4, 4)
    _plot_target_data(reduction, target_data, target, target_type)
    
    if target_type == 'pseudotime':
        target_name = "'s pseudotime"
    elif target == ALL_CELLS:
        target_name = ' Identities'
    else:
        target_name = "'s identity"

    title = f'Predicting {target}{target_name} using {set_name}'
    fontsize = (10 if len(title) < 90 else 7) if not as_single_row else 14
    plt.suptitle(title, fontsize=fontsize)

    save_plot(f'predicting {target} using {set_name}', os.path.join(output, 'pathways', target_type) if format else None, format=format)    


def plot_all_cell_types_and_trajectories(
        reduction: pd.DataFrame, 
        cell_types: pd.DataFrame | None,
        pseudotime: pd.DataFrame | None,
        output: str = None,
        subtitle: bool = False,
        format: str = 'png',
    ):
    num_plots = int(cell_types is not None) + int(pseudotime is not None)
    plt.figure(figsize=(6 * num_plots, 5), dpi=DPI)

    if cell_types is not None:
        plt.subplot(1, num_plots, 1)
        _plot_cell_types(reduction, cell_types, title=True, subtitle=subtitle)
    
    if pseudotime is not None:
        plt.subplot(1, num_plots, num_plots)
        _plot_pseudotime(reduction, pseudotime, title=True, subtitle=subtitle)
    
    save_plot('targets', output, format=format)


def plot(
        output: str,
        expression: pd.DataFrame | str = 'expression', 
        reduction: pd.DataFrame | str = 'reduction', 
        cell_types: pd.DataFrame | str = 'cell_types',
        pseudotime: pd.DataFrame | str = 'pseudotime',
        classification_results: pd.DataFrame | str = 'cell_type_classification',
        regression_results: pd.DataFrame | str = 'pseudotime_regression',
        threshold: float = THRESHOLD,
        top: int = None,
        all: bool = False,
    ):
    """
    top: number of top pathways to plot for each target
    all: whether to plot all pathways
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

        heatmap_pathways, exp_plot_pathways = [], []

        if all or data.shape[0] <= MAP_SIZE:  # plot all pathways
            heatmap_pathways = data.index
            for target in data.columns:
                for pathway_name in heatmap_pathways:
                    plot_experiment(output, target, pathway_name, target_type, results, target_data, expression, reduction)

        else:  # plot interesting pathways
            size = max(MAP_SIZE // data.shape[1], 1)
            # heatmap_pathways.extend(get_top_sum_pathways(data, ascending=True, size=3))
            non_unique_targets = []
            for target in data.columns:
                if target != ALL_CELLS:
                    pathway_names = get_column_unique_pathways(data, target, top if top else size, threshold)
                    if not pathway_names:
                        non_unique_targets.append(target)
                    heatmap_pathways.extend(pathway_names[:size])
                    for pathway_name in pathway_names:
                        exp_plot_pathways.append((target, pathway_name))
                        plot_experiment(output, target, pathway_name, target_type, results, target_data, expression, reduction)
            # heatmap_pathways.extend(get_top_sum_pathways(data, ascending=False, size=3))
            data = data.drop(non_unique_targets, axis=1)

        if len(heatmap_pathways) < data.shape[0]:
            plot_p_values(data, cluster_rows=True, title=f'{target_type} Prediction using All Pathways', output=output)
        plot_p_values(data.loc[heatmap_pathways], title=f'{target_type} Prediction', output=output)
        
        save_csv(pd.DataFrame(exp_plot_pathways, columns=[TARGET_COL, 'pathway']), title=f'top_{target_type}_pathways', output_path=output, keep_index=False)

        del data
        del results
        del heatmap_pathways
        del exp_plot_pathways
