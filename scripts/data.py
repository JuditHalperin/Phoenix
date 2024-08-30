import warnings, random
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
        min_cell_percent: float = 5,
        min_lineage_percent: float = 5,
        last_cells: int = 50,
        last_survived_cells: int = 30,
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
    min_cell_percent: minimum percentage of single cells required to keep a cell type
    min_lineage_percent: minimum percentage of single cells required to keep a lineage
    last_cells: number of last cells in each trajectory considered as important for the trajectory
    last_survived_cells: minimum number of last cells required to remain after cell filtering in order to keep a lineage
    """

    # Filter and normalize
    expression = preprocess_expression(expression, preprocessed, verbose=verbose)

    # Exclude targets
    if cell_types is not None:
        cell_types = cell_types.loc[expression.index]

        # Remove rare cell types
        rare_cell_types = [cell_type for cell_type in cell_types[CELL_TYPE_COL].unique()
                           if (cell_types[CELL_TYPE_COL] == cell_type).sum() / cell_types.shape[0] * 100 < min_cell_percent] if min_cell_percent else []
        
        # Remove requested to exclude cell types
        exclude_cell_types = [cell_type for cell_type in exclude_cell_types
                              if cell_type in cell_types[CELL_TYPE_COL].tolist()] if exclude_cell_types else []
        
        exclude_cell_types = list(set(exclude_cell_types + rare_cell_types))
        if exclude_cell_types:
            if verbose:
                print(f'Excluding cell types: {", ".join(exclude_cell_types)}...')
            cell_types = cell_types[~cell_types[CELL_TYPE_COL].isin(exclude_cell_types)]
            expression = expression.loc[cell_types.index]

    if pseudotime is not None:
        original_pseudotime = pseudotime.copy()
        pseudotime = pseudotime.loc[expression.index]

        # Remove lineages with too few last cells survived after cell type filtering
        removed_lineages = []
        if last_cells >= last_survived_cells:
            for lineage in pseudotime.columns:
                original_last_cells = original_pseudotime[lineage].dropna().sort_values(ascending=False).head(last_cells).index
                if original_last_cells.isin(pseudotime.index).sum() < last_survived_cells:
                    removed_lineages.append(lineage)
            
        # Remove too short lineages (many NA values)
        short_lineages = [lineage for lineage in pseudotime.columns
                          if pseudotime[lineage].isna().sum() / pseudotime.shape[0] * 100 > min_lineage_percent] if min_lineage_percent else []
        
        # Remove requested to exclude lineages
        exclude_lineages = [lineage for lineage in exclude_lineages
                            if lineage in pseudotime.columns] if exclude_lineages else []
        
        exclude_lineages = list(set(exclude_lineages + short_lineages + removed_lineages))
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
    return transform_log(re_transform_log(gene_set_expression).sum(axis=1))
