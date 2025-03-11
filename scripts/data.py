import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import random
import pandas as pd
import numpy as np
import scanpy as sc
from scripts.consts import ALL_CELLS, CELL_TYPE_COL, NUM_GENES, SEED
from scripts.utils import transform_log, re_transform_log
from scripts.output import save_csv

sc.settings.verbosity = 0


def preprocess_expression(expression: pd.DataFrame, preprocessed: bool, num_genes: int = NUM_GENES, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print('Running single-cell preprocessing...')
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


def reduce_dimension(expression: pd.DataFrame, reduction_method: str, seed: int) -> pd.DataFrame:
    print('Reducing single-cell dimensionality...')

    adata = sc.AnnData(expression)

    sc.tl.pca(adata, random_state=seed)
    if reduction_method == 'umap':
        sc.pp.neighbors(adata, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
    elif reduction_method == 'tsne':
        sc.tl.tsne(adata, random_state=seed)

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
        exclude_cell_types: list[str] = [],
        exclude_lineages: list[str] = [],
        seed: int = SEED,
        output: str = None,
        verbose: bool = True,
    ):
    """
    expression: single-cell raw expression data
    cell_types: single-cell cell type annotations
    pseudotime: single-cell pseudotime values
    reduction: reduction coordinates or method name to use for dimensionality reduction
    preprocessed: whether expression data are already filtered and log-normalized. In this case neither cell filtering nor normalization is applied, only gene filtering if necessary
    exclude_cell_types: list of cell types to exclude from the analysis
    exclude_lineages: list of lineages to exclude from the analysis
    """

    # Filter and normalize
    expression = preprocess_expression(expression, preprocessed, verbose=verbose)

    # Exclude targets
    if cell_types is not None:
        cell_types = cell_types.loc[expression.index]
        exclude_cell_types = [cell_type for cell_type in exclude_cell_types
                              if cell_type in cell_types[CELL_TYPE_COL].tolist()] if exclude_cell_types else []
        if exclude_cell_types:
            if verbose:
                print(f'Excluding cell types: {", ".join(exclude_cell_types)}...')
            cell_types = cell_types[~cell_types[CELL_TYPE_COL].isin(exclude_cell_types)]
            expression = expression.loc[cell_types.index]

    if pseudotime is not None:
        pseudotime = pseudotime.loc[expression.index]
        exclude_lineages = [lineage for lineage in exclude_lineages
                            if lineage in pseudotime.columns] if exclude_lineages else []
        if exclude_lineages:
            if verbose:
                print(f'Excluding lineages: {", ".join(exclude_lineages)}...')
            pseudotime.drop(columns=exclude_lineages, inplace=True)

    # Reduce dimensions
    if isinstance(reduction, str):
        reduction = reduce_dimension(expression, reduction, seed)
    reduction = reduction.loc[expression.index]

    # Save preprocessed data
    if output:
        save_csv(expression, 'expression', output)
        save_csv(cell_types, 'cell_types', output)
        save_csv(pseudotime, 'pseudotime', output)
        save_csv(reduction, 'reduction', output)

    return expression, cell_types, pseudotime, reduction


def get_cell_types(cell_types: pd.DataFrame) -> list[str]:
    cell_type_list = (cell_types[CELL_TYPE_COL].unique().tolist() + [ALL_CELLS]) if cell_types is not None else []
    random.shuffle(cell_type_list)
    return cell_type_list


def get_lineages(pseudotime: pd.DataFrame) -> list[str]:
    lineage_list = pseudotime.columns.tolist() if pseudotime is not None else []
    random.shuffle(lineage_list)
    return lineage_list


def sum_gene_expression(gene_set_expression: pd.DataFrame) -> pd.Series:
    untransformed = re_transform_log(gene_set_expression)
    summed = untransformed.sum() if untransformed.ndim == 1 else untransformed.sum(axis=1)
    return transform_log(summed)
