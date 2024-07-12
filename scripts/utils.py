import re, os, yaml, time
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.consts import SIZES, TARGET_COL


transform_log = lambda x: np.log2(x + 1)
re_transform_log = lambda x: 2 ** x - 1


def get_full_path(path: str) -> str:
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    return os.path.abspath(path)


def make_valid_filename(filename: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '', filename.replace(' ', '_'))


def make_valid_term(term: str) -> str:
    return term.replace(',', '')


def convert2sci(num):
    return '{:.0e}'.format(num)


def convert2str(info: str | list | dict | None):
    if isinstance(info, list):
        return ', '.join([convert2str(i) for i in info])
    if isinstance(info, dict):
        return ', '.join(f"{convert2str(k)}: {convert2str(v)}" for k, v in info.items())
    return str(info)    


def define_task(cell_type: str = None, lineage: str = None):
    if lineage:
        return f'regression_{lineage}'
    if cell_type:
        return f'classification_{cell_type}'
    raise RuntimeError()


def define_background(set_size: int, repeats: int, cell_type: str = None, lineage: str = None):
    return f'{define_task(cell_type, lineage)}_size{set_size}_repeats{repeats}'


def define_set_size(set_len: int, set_fraction: float, min_set_size: int) -> int:
    set_size = min(max(int(set_len * set_fraction), min_set_size), set_len)
    return min(SIZES, key=lambda x: abs(x - set_size))


def get_color_mapping(cell_types: list[str]) -> dict[str, str]:
    """
    cell_types: unique
    """
    color_palette = sns.color_palette('Set1', n_colors=len(cell_types))
    return {cell_type: color_palette[i] for i, cell_type in enumerate(cell_types)}


def remove_outliers(values: list[float]) -> list[float]:
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [i for i in values if i >= lower_bound and i <= upper_bound]


def read_csv(path: str, index_col: int = 0) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=index_col)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path '{path}'")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{path}' is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Parsing failed for file '{path}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading '{path}': {str(e)}")


def save_csv(data: list[dict] | pd.DataFrame, title: str, output_path: str, keep_index: bool = True) -> None:
    if not data: return
    data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    data.to_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), index=keep_index)


def load_background_scores(background: str, cache_path: str = None):
    if cache_path and os.path.exists(f'{cache_path}/{background}.yml'):
        print(f'Loading background {background} from cache...')
        with open(f'{cache_path}/{make_valid_filename(background)}.yml', 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    return None


def save_background_scores(background_scores: list[float], background: str, cache_path: str):
    print(f'Saving background {background} in cache...')
    with open(f'{cache_path}/{make_valid_filename(background)}.yml', 'w') as file:
        yaml.dump(background_scores, file)


def summarise_result(target, set_name, original_gene_set, gene_set, top_genes, set_size, feature_selection, predictor, metric, cross_validation, repeats, seed, pathway_score, background_scores: list[float], p_value):
    result = {
        TARGET_COL: target,
        'set_name': set_name,
        'original_gene_set': original_gene_set,
        'gene_set': gene_set,
        'top_genes': top_genes,
        'set_size': set_size,
        'feature_selection': feature_selection,
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
    return {key: convert2str(value) for key, value in result.items()}


def read_results(title: str, output_path: str, index_col=None) -> pd.DataFrame:
    return read_csv(os.path.join(output_path, f'{make_valid_filename(title)}.csv'), index_col=index_col)


def get_preprocessed_data(data: pd.DataFrame | str, output_path: str):
    if isinstance(data, str):
        data = read_results(data, output_path, index_col=0)
    return data


def get_experiment(results: pd.DataFrame | str, output_path: str, set_name: str = None, target: str = None):
    if isinstance(results, str):
        results = read_results(results, output_path)
    if set_name:
        results = results[results['set_name'] == set_name]
    if target:
        results = results[results[TARGET_COL] == target]
    return results


def save_plot(title: str, output: str = None):
    plt.tight_layout()
    if output:
        plt.savefig(f'{output}/{make_valid_filename(title)}.png')
    else:
        plt.show()
    plt.close()


def show_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(f'Running {func.__name__} took {f"{minutes:02d}:{seconds:02d}"} minutes to run.')
        return result
    return wrapper
