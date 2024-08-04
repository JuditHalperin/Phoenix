import os, yaml, glob
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scripts.consts import TARGET_COL
from scripts.utils import make_valid_filename, convert_to_str, convert_from_str, adjust_p_value


def read_csv(path: str, index_col: int = 0, dtype=None, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f'Reading file at {path}...')
    try:
        return pd.read_csv(path, index_col=index_col, dtype=dtype)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path '{path}'")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{path}' is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Parsing failed for file '{path}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading '{path}': {str(e)}")


def save_csv(data: list[dict] | pd.DataFrame, title: str, output_path: str, keep_index: bool = True) -> None:
    if data is None: return
    data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    # TODO: if already exists
    data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), index=keep_index)


def load_background_scores(background: str, cache_path: str = None, verbose: bool = False):
    background = make_valid_filename(background).lower()
    if cache_path and os.path.exists(f'{cache_path}/{background}.yml') and os.path.getsize(f'{cache_path}/{background}.yml') > 0:
        if verbose:
            print(f'Loading background {background} from cache...')
        with open(f'{cache_path}/{background}.yml', 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    return []


def save_background_scores(background_scores: list[float], background: str, cache_path: str = None, verbose: bool = False):
    if cache_path:
        background = make_valid_filename(background).lower()
        if verbose:
            print(f'Saving background {background} in cache...')
        with open(f'{cache_path}/{background}.yml', 'w') as file:
            yaml.dump(background_scores, file)


def read_gene_sets(output_path: str) -> dict[str, list[str]]:
    df = read_csv(output_path, index_col=False, dtype=str)
    return {column: df[column].dropna().tolist() for column in df.columns}


def save_gene_sets(gene_sets: dict[str, list[str]], output_path: str, by_set: bool = False) -> None:
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gene_sets.items()]))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df.to_csv(f'{output_path}/gene_sets.csv', index=False)

    if by_set:
        for col in df.columns:
            pd.DataFrame(df[col]).dropna().to_csv(f'{output_path}/{make_valid_filename(col)}.csv', index=False)


def summarise_result(target, set_name, top_genes, set_size, feature_selection, predictor, metric, cross_validation, repeats, seed, pathway_score, background_scores: list[float], p_value):
    result = {
        TARGET_COL: target,
        'set_name': set_name,
        'top_genes': top_genes,
        'set_size': set_size,
        'feature_selection': feature_selection if feature_selection else 'None',
        'predictor': predictor,
        'metric': metric,
        'cross_validation': cross_validation,
        'repeats': repeats,
        'seed': seed,
        'pathway_score': pathway_score,
        'background_scores': background_scores,
        'background_score_mean': np.mean(background_scores),
        'p_value': p_value,
    }
    return {key: convert_to_str(value) for key, value in result.items()}


def read_results(title: str, output_path: str, index_col=None) -> pd.DataFrame | None:
    try:
        title = f'{title}.csv' if '.csv' not in title else title
        return read_csv(os.path.join(output_path, f'{make_valid_filename(title)}'), index_col=index_col)
    except Exception as e:
        print(e)
        return None


def get_preprocessed_data(data: pd.DataFrame | str, output_path: str):
    if isinstance(data, str):
        data = read_results(data, output_path, index_col=0)
    return data


def aggregate_result(result_type: str, output: str, tmp: str):
    df = read_results(result_type, output)
    if df is not None:
        if 'fdr' not in df.columns:
            df['fdr'] = adjust_p_value(df['p_value'])
        return df
    
    dfs = []
    for path in glob.glob(os.path.join(tmp, f'{result_type}_batch*.csv')):
        df = read_results(os.path.basename(path), tmp, index_col=None)
        if df is not None:
            dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)

    dfs['fdr'] = adjust_p_value(dfs['p_value'])
    dd.from_pandas(dfs, npartitions=1).to_csv(os.path.join(output, f'{result_type}.csv'), single_file=True, index=False)
    return dfs


def get_experiment(results: pd.DataFrame | str, output_path: str, set_name: str = None, target: str = None) -> pd.DataFrame | dict:
    if isinstance(results, str):
        results = read_results(results, output_path)
    if set_name and results is not None:
        results = results[results['set_name'] == set_name]
    if target and results is not None:
        results = results[results[TARGET_COL] == target]
    
    if set_name and target and results is not None:
        results = results.iloc[0]
        return {key: convert_from_str(results[key]) for key in results.index}

    return results


def save_plot(title: str, output: str = None):
    plt.tight_layout()
    if output:
        plt.savefig(f'{output}/{make_valid_filename(title)}.png')
    else:
        plt.show()
    plt.close()
