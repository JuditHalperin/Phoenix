import re, os, yaml, time, glob
from functools import wraps
from argparse import Namespace
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scripts.consts import SIZES, TARGET_COL, LIST_SEP


transform_log = lambda x: np.log2(x + 1)
re_transform_log = lambda x: 2 ** x - 1


def get_full_path(path: str) -> str:
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    return os.path.abspath(path)


def make_valid_filename(filename: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.]+', '', filename.replace(' ', '_')).lower()[:100]


def make_valid_term(term: str) -> str:
    return term.replace(',', '')


def convert2sci(num: float) -> str:
    return '{:.0e}'.format(num)


def convert_to_str(info: str | list | dict | None) -> str:
    if isinstance(info, list):
        return LIST_SEP.join([convert_to_str(i) for i in info])
    if isinstance(info, dict):
        return LIST_SEP.join(f'{convert_to_str(k)}: {convert_to_str(v)}' for k, v in info.items())
    return str(info)


def convert_from_str(info: str) -> list | float | str:
    if isinstance(info, str) and LIST_SEP in info:
        return [convert_from_str(i) for i in info.split(LIST_SEP)]
    try:
        return float(info)
    except:
        return info


def define_task(cell_type: str = None, lineage: str = None):
    if lineage:
        return f'regression_{lineage}'
    if cell_type:
        return f'classification_{cell_type}'
    raise RuntimeError()


def define_background(set_size: int, repeats: int, cell_type: str = None, lineage: str = None):
    # TODO: add info such as model name
    return f'{define_task(cell_type, lineage)}_size{set_size}_repeats{repeats}'


def define_set_size(set_len: int, set_fraction: float, min_set_size: int) -> int:
    set_size = min(max(int(set_len * set_fraction), min_set_size), set_len)
    return max((x for x in SIZES if x <= set_size), default=None)


def define_batch_size(gene_set_len: int, processes: int) -> int:
    if not processes:
        return gene_set_len
    return int(np.ceil(gene_set_len / processes))


def get_gene_set_batches(gene_sets: list[str], batch_size: int) -> list[dict[str, list[str]]]:
    gene_set_batches = []
    for batch_start in range(0, len(gene_sets), batch_size):
        set_names = list(gene_sets.keys())[batch_start:min(batch_start + batch_size, len(gene_sets))]
        gene_set_batches.append({set_name: gene_sets[set_name] for set_name in set_names})
    return gene_set_batches


def get_gene_set_batch(gene_sets: dict[str, list[str]], batch: int | None = None, batch_size: int | None = None) -> dict[str, list[str]]:
    """
    batch: number between 1 and `processes`, or None for a single batch
    """
    if batch is None:
        return gene_sets
    batch_start = (batch - 1) * batch_size
    batch_end = min(batch_start + batch_size, len(gene_sets))
    set_names = list(gene_sets.keys())[batch_start:batch_end]
    return {set_name: gene_sets[set_name] for set_name in set_names}


def parse_missing_args(args):
    args_dict = vars(args)
    updated_args = {k: (None if v == 'None' else v) for k, v in args_dict.items()}
    return Namespace(**updated_args)


def _estimate_mem(task_len: int) -> str:
    # TODO: expand estimation
    return f'{max(task_len // 7, 3)}G'


def _estimate_time(task_len: int) -> str:
    # TODO: expand estimation
    return '15:0:0' if task_len > 30 else '1:30:0'


def get_batch_run_cmd(processes: int | None, batch_size: int, task_len: int, **kwargs) -> str:

    batch_args = ' '.join([f'--{k}={v}' for k, v in kwargs.items()]) + f' --batch_size={batch_size}'
    batch_args += ' --batch \$SLURM_ARRAY_TASK_ID' if processes else ''
    excute_cmd = f'python run_batch.py {batch_args}'
  
    if processes:
        report_path = kwargs.get('tmp')
        return (
            f'sbatch --job-name=batch_run '
            f'--mem={_estimate_mem(task_len)} '
            f'--time={_estimate_time(task_len)} '
            f'--array=1-{processes} '
            f'--output={report_path}/%A_%a_batch_run.out '
            f'--error={report_path}/%A_%a_batch_run.err '
            f'--wrap=\"{excute_cmd}\"'
        )

    return excute_cmd


def get_aggregation_cmd(output: str, tmp: str | None, job_id: int, process: int) -> str:
    aggregate_cmd = (
        f"python -c 'from run import summarize; "
        rf"summarize(\"{output}\", \"{tmp}\")'"
    )
    report_path = tmp if tmp else output
    sbatch_cmd = (
        f"sbatch --job-name=aggregation "
        f"--mem=10G --time=1:0:0 "
        f"--output={report_path}/%j_aggregation.out "
        f"--error={report_path}/%j_aggregation.err "
        f"--wrap=\"{aggregate_cmd}\" "
    )
    if job_id:
        sbatch_cmd += f"--dependency=afterok:{','.join([f'{job_id}_{p + 1}' for p in range(process)])} "
    return sbatch_cmd


# def get_cmd(func: str, args: dict[str, str], script: str, processes: int, sbatch: bool = False, mem: str = '1G', time: str = '0:30:0', report_path: str = None, last_job_id: str = None):

#     args = ', '.join([rf'{k}={repr(v)}' if isinstance(v, str) else rf'{k}={v}' for k, v in args.items()])
#     python_cmd = (
#         f"python -c 'from scripts.{script} import {func}; "
#         rf"{func}({args})'"
#     )
#     if not sbatch:
#         return python_cmd
    
#     report_info = '%j' if not processes else r'%A_%a'
#     sbatch_cmd = (
#         f"sbatch --job-name={func} --mem={mem} --time={time} "
#         f"--output={report_path}/{report_info}_{func}.out "
#         f"--error={report_path}/{report_info}_{func}.err "
#         f"--wrap=\"{python_cmd}\" "
#     )
   

def get_color_mapping(cell_types: list[str]) -> dict[str, str]:
    """
    cell_types: unique
    """
    color_palette = sns.color_palette('Paired', n_colors=len(cell_types))
    return {cell_type: color_palette[i] for i, cell_type in enumerate(cell_types)}


def remove_outliers(values: list[float]) -> list[float]:
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [i for i in values if i >= lower_bound and i <= upper_bound]


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


def adjust_p_value(p_values):
    return multipletests(p_values, method='fdr_bh')[1]


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

    # ddf = dd.read_csv(os.path.join(tmp, f'{result_type}_batch*.csv'))
    # df = ddf.compute()
    # df['fdr'] = adjust_p_value(df['p_value'])
    # ddf = dd.from_pandas(df, npartitions=1)
    # ddf.to_csv(os.path.join(output, f'{result_type}.csv'), single_file=True, index=False)
    # return ddf
    dfs['fdr'] = adjust_p_value(dfs['p_value'])
    dd.from_pandas(dfs, npartitions=1).to_csv(os.path.join(output, f'{result_type}.csv'), single_file=True, index=False)
    return dfs



# def aggregate_results(output: str, tmp: str):

#     def aggregate_result(result_type: str):
#         df = read_results(result_type, output)
#         if df is not None:
#             df['fdr'] = adjust_p_value(df['p_value'])
#             return df
        
#         ddf = dd.read_csv(os.path.join(tmp, f'{result_type}_batch*.csv'))
#         df = ddf.compute()
#         df['fdr'] = adjust_p_value(df['p_value'])
#         ddf = dd.from_pandas(df, npartitions=1)
#         ddf.to_csv(os.path.join(output, f'{result_type}.csv'), single_file=True, index=False)
#         return ddf
    
#     results = aggregate_result('cell_type_classification'), aggregate_result('pseudotime_regression')

#     from scripts.visualization import plot
#     plot(output)

#     return results


def get_preprocessed_data(data: pd.DataFrame | str, output_path: str):
    if isinstance(data, str):
        data = read_results(data, output_path, index_col=0)
    return data


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


def show_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)        
        if seconds > 5:
            info = f" on data with shape {kwargs.get('X').shape} using {kwargs.get('predictor').__name__}" if kwargs.get('X') is not None and kwargs.get('predictor') is not None else ""
            print(f'Running {func.__name__}{info} took {f"{minutes:02d}:{seconds:02d}"} minutes to run.')
        return result
    return wrapper
