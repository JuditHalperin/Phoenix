import warnings, random
import pandas as pd
import numpy as np
import scanpy as sc
from scripts.consts import ALL_CELLS, CELL_TYPE_COL, NUM_GENES, CELL_PERCENT
from scripts.utils import transform_log, re_transform_log
from scripts.output import save_csv

sc.settings.verbosity = 0


def preprocess_expression(expression: pd.DataFrame, preprocessed: bool, num_genes: int = NUM_GENES, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print('Running single-cell preprocessing...')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        adata = sc.AnnData(expression)

    if not preprocessed:
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=5)
        sc.pp.filter_genes(adata, min_counts=500)
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Filter genes using top mean count
    if len(adata.var) > num_genes:
        adata.var['mean_counts'] = adata.X.mean(axis=0)
        adata = adata[:, adata.var_names[np.argsort(adata.var['mean_counts'])[::-1]][:num_genes]]

    return pd.DataFrame(data=adata.X, index=adata.obs_names, columns=adata.var_names)


def reduce_dimension(expression: pd.DataFrame, reduction_method: str) -> pd.DataFrame:
    print('Reducing single-cell dimensionality...')

    adata = sc.AnnData(expression)

    sc.tl.pca(adata)
    if reduction_method == 'umap':
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    elif reduction_method == 'tsne':
        sc.tl.tsne(adata)

    return pd.DataFrame(
        adata.obsm[f'X_{reduction_method}'],
        columns=[f'{reduction_method}1', f'{reduction_method}2'],
        index=adata.obs_names
    )


def preprocess_data(
        expression: pd.DataFrame,
        cell_types: pd.DataFrame,
        pseudotime: pd.DataFrame,
        reduction: pd.DataFrame | str,
        preprocessed: bool = False,
        min_cell_percent: float = CELL_PERCENT,
        exclude_cell_types: list[str] = [],
        exclude_lineages: list[str] = [],
        output: str = None,
        verbose: bool = True,
    ):
    """
    preprocessed: whether expression data are already filtered and log-normalized. In this case neither cell filtering nor normalization is applied, only gene filtering if necessary
    """

    # Filter and normalize
    expression = preprocess_expression(expression, preprocessed, verbose=verbose)

    # Exclude targets
    if cell_types is not None:
        cell_types = cell_types.loc[expression.index]
        rare_cell_types = [cell_type for cell_type in cell_types[CELL_TYPE_COL].unique() if (cell_types[CELL_TYPE_COL] == cell_type).sum() / cell_types.shape[0] * 100 < min_cell_percent]
        exclude_cell_types = [cell_type for cell_type in exclude_cell_types if cell_type in cell_types[CELL_TYPE_COL].tolist()] if exclude_cell_types else []
        exclude_cell_types = list(set(exclude_cell_types + rare_cell_types))
        if exclude_cell_types:
            if verbose:
                print(f'Excluding cell types: {", ".join(exclude_cell_types)}...')
            cell_types = cell_types[~cell_types[CELL_TYPE_COL].isin(exclude_cell_types)]
            expression = expression.loc[cell_types.index]

    if pseudotime is not None:
        pseudotime = pseudotime.loc[expression.index]
        # TODO: remove if too few cells
        exclude_lineages = [lineage for lineage in exclude_lineages if lineage in pseudotime.columns] if exclude_lineages else []
        if exclude_lineages:
            if verbose:
                print(f'Excluding lineages: {", ".join(exclude_lineages)}...')
            pseudotime.drop(columns=exclude_lineages)

    # Reduce dimensions
    if isinstance(reduction, str):
        reduction = reduce_dimension(expression, reduction)
    reduction = reduction.loc[expression.index]

    # Save preprocessed data
    if output:
        save_csv(expression, 'expression', output)
        save_csv(cell_types, 'cell_types', output)
        save_csv(pseudotime, 'pseudotime', output)
        save_csv(reduction, 'reduction', output)

    return expression, cell_types, pseudotime, reduction


def intersect_genes(gene_set: list[str], all_genes: list[str], required_len: int = 5) -> list[str]:

    is_set = lambda gene_set: len(list(set(gene_set).intersection(set(all_genes)))) >= min(required_len, len(gene_set) // 2)
    intersect_set = lambda gene_set: sorted([g for g in set(gene_set) if g in all_genes])

    if is_set(gene_set):
        return intersect_set(gene_set)

    gene_set = [g.lower() for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    gene_set = [g.upper() for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    to_title = lambda word: word[0].upper() + word[1:].lower()
    gene_set = [to_title(g) for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    return []


def get_cell_types(cell_types: pd.DataFrame) -> list[str]:
    cell_type_list = (cell_types[CELL_TYPE_COL].unique().tolist() + [ALL_CELLS]) if cell_types is not None else []
    random.shuffle(cell_type_list)
    return cell_type_list


def get_lineages(pseudotime: pd.DataFrame) -> list[str]:
    lineage_list = pseudotime.columns.tolist() if pseudotime is not None else []
    random.shuffle(lineage_list)
    return lineage_list


def get_top_sum_pathways(data, ascending: bool, size: int) -> list[str]:
    return data.copy().dropna(axis=0).sum(axis=1).sort_values(ascending=ascending).head(size).index.tolist()


def get_column_unique_pathways(data, col: str, size: int, threshold: float) -> list[str]:
    tmp = data.copy()
    tmp = tmp[(tmp[col] == tmp.min(axis=1)) & (tmp[col] <= threshold if threshold else 1)]
    tmp = tmp.loc[tmp[col].dropna().sort_values(ascending=True).index[:200]]
    to_drop = [col, ALL_CELLS] if ALL_CELLS in tmp.columns else [col]
    tmp['max_diff'] = tmp.drop(to_drop, axis=1).min(axis=1) - tmp[col]
    tmp = tmp.sort_values(by='max_diff', ascending=False)
    return tmp.head(size).index.tolist()


def get_all_column_unique_pathways(data, size: int, threshold: float):
    size = size // data.shape[1]
    pathways = []
    for col in data.columns:
        if col != ALL_CELLS:
            pathways.extend(get_column_unique_pathways(data, col, size, threshold))
    return pathways


def sum_gene_expression(gene_set_expression: pd.DataFrame) -> pd.Series:
    return transform_log(re_transform_log(gene_set_expression).sum(axis=1))
