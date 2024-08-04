import os
from scripts.utils import estimate_mem, estimate_time


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


def get_batch_run_cmd(processes: int | None, batch_size: int, task_len: int, **kwargs) -> str:

    batch_args = ' '.join([f'--{k}={v}' for k, v in kwargs.items()]) + f' --batch_size={batch_size}'
    batch_args += ' --batch \$SLURM_ARRAY_TASK_ID' if processes else ''
    excute_cmd = f'python run_batch.py {batch_args}'
  
    if processes:
        report_path = kwargs.get('tmp')
        return (
            f'sbatch --job-name=batch_run '
            f'--mem={estimate_mem(task_len)} '
            f'--time={estimate_time(task_len)} '
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


def get_cmd(
        func: str,
        args: dict[str, str],
        script: str,
        sbatch: bool = False,
        processes: int = None,
        array_param: str = 'batch',
        mem: str = '1G',
        time: str = '0:30:0',
        report_path: str = None,
        previous_job_id: str = None,
        previous_processes: int = None,
    ):

    args = ', '.join([rf'{k}={repr(v)}' if isinstance(v, str) else rf'{k}={v}' for k, v in args.items()])
    args += f', {array_param}=\$SLURM_ARRAY_TASK_ID' if processes else ''
    
    script = f'scripts.{script}' if not os.path.exists(f'{script}.py') else script
    python_cmd = (
        f"python -c 'from {script} import {func}; "
        rf"{func}({args})'"
    )
    if not sbatch:
        return python_cmd
    
    report_info = '%j' if not processes else r'%A_%a'
    sbatch_cmd = (
        f"sbatch --job-name={func} --mem={mem} --time={time} "
        f"--output={report_path}/{report_info}_{func}.out "
        f"--error={report_path}/{report_info}_{func}.err "
        f"--wrap=\"{python_cmd}\" "
    )

    sbatch_cmd += f'--array=1-{processes} ' if processes else ''

    if previous_job_id:
        if previous_processes:
            sbatch_cmd += f"--dependency=afterok:{','.join([f'{previous_job_id}_{p + 1}' for p in range(previous_processes)])} "
        else:
            sbatch_cmd += f"--dependency=afterok:{previous_job_id} "
    
    return sbatch_cmd
