import os, subprocess
from scripts.utils import get_file_size


def get_cmd(
        func: str,
        args: dict[str, str],
        script: str = 'run',
        sbatch: bool = False,
        processes: int = None,
        mem: str = '1G',
        time: str = '0:30:0',
        report_path: str = None,
        previous_job_id: str = None,
        previous_processes: int = None,
    ):

    parsed_args = ', '.join([f'{k}={repr(v) if isinstance(v, str) else v}' for k, v in args.items()])
    parsed_args = parsed_args.replace("'", '\\"')
    
    script = f'scripts.{script}' if not os.path.exists(f'{script}.py') else script
    python_cmd = (
        f"python -c 'from {script} import {func}; "
        f"{func}({parsed_args})' "
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


def execute_cmd(cmd, title: str, processes: int = None) -> int:
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:  # if sbatch run
        job_id = process.stdout.strip().split()[-1]
        print(f'Executing {title} as job {job_id}' + (f' ({processes} processes)' if processes else '') + '...')
        return job_id
    except:
        raise ValueError('Currently not supporting non-sbatch commands')


# TODO: estimate memory and time for each step

def run_setup_cmd(args: dict, tmp: str = None) -> str:
    cmd = get_cmd(
        func='setup',
        args=args,
        script='run',
        sbatch=True,
        mem='5G',
        time='0:15:0',
        report_path=tmp,
    )
    return execute_cmd(cmd, 'initial setup')


def run_experiments_cmd(setup_job_id: int, args: dict, tmp: str = None) -> int:
    cmd = get_cmd(
        func='run_experiments', 
        args=args,
        script='run',
        sbatch=True,
        processes=args['processes'],
        mem='10G',
        time='15:0:0',
        report_path=tmp,
        previous_job_id=setup_job_id,
    )
    return execute_cmd(cmd, 'experiments', args['processes'])


def run_aggregation_cmd(exp_job_id: int, exp_processes: int | None, output: str, tmp: str) -> int:
    cmd = get_cmd(
        func='summarize',
        args={'output': output, 'tmp': tmp},
        script='run',
        sbatch=True,
        mem='5G',  
        time='0:15:0',
        report_path=tmp,
        previous_job_id=exp_job_id,
        previous_processes=exp_processes,
    )
    return execute_cmd(cmd, 'aggregation and plotting')
