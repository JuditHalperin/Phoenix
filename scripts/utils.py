import re, os, time
from functools import wraps
from argparse import Namespace
import numpy as np
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scripts.consts import SIZES, LIST_SEP


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


def convert_to_sci(num: float) -> str:
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


def adjust_p_value(p_values):
    return multipletests(p_values, method='fdr_bh')[1]


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
