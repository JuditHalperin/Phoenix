import unittest
from tests.interface import Test
from scripts.computation import get_gene_set_batch, get_cmd


class BatchTest(Test):

    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
                    
    def test_get_gene_set_batch(self):
        self.assertEqual(get_gene_set_batch(self.gene_sets), self.gene_sets)
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=3), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=2, batch_size=3), {'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=3, batch_size=2), {'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=4), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4']})


class CmdTest(Test):

    def test_non_sbatch_command(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script'
        )
        expected_cmd = "python -c 'from scripts.my_script import my_function; my_function(arg1='value1', arg2=2)'"
        self.assertEqual(cmd, expected_cmd)
    
    def test_sbatch_command_without_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            report_path='/path/to/report'
        )
        expected_cmd = (
            "sbatch --job-name=my_function --mem=1G --time=0:30:0 "
            "--output=/path/to/report/%j_my_function.out "
            "--error=/path/to/report/%j_my_function.err "
            "--wrap=\"python -c 'from scripts.my_script import my_function; my_function(arg1='value1', arg2=2)'\" "
        )
        self.assertEqual(cmd, expected_cmd)

    def test_sbatch_command_with_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            processes=5,
            report_path='/path/to/report'
        )
        expected_cmd = (
            "sbatch --job-name=my_function --mem=1G --time=0:30:0 "
            "--output=/path/to/report/%A_%a_my_function.out "
            "--error=/path/to/report/%A_%a_my_function.err "
            "--wrap=\"python -c 'from scripts.my_script import my_function; my_function(arg1='value1', arg2=2, batch=\$SLURM_ARRAY_TASK_ID)'\" "
            "--array=1-5 "
        )
        self.assertEqual(cmd, expected_cmd)

    def test_sbatch_command_with_dependency(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            report_path='/path/to/report',
            previous_job_id='12345'
        )
        expected_cmd = (
            "sbatch --job-name=my_function --mem=1G --time=0:30:0 "
            "--output=/path/to/report/%j_my_function.out "
            "--error=/path/to/report/%j_my_function.err "
            "--wrap=\"python -c 'from scripts.my_script import my_function; my_function(arg1='value1', arg2=2)'\" "
            "--dependency=afterok:12345 "
        )
        self.assertEqual(cmd, expected_cmd)
    
    def test_sbatch_command_with_dependency_and_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            processes=5,
            report_path='/path/to/report',
            previous_job_id='12345',
            previous_processes=3
        )
        expected_cmd = (
            "sbatch --job-name=my_function --mem=1G --time=0:30:0 "
            "--output=/path/to/report/%A_%a_my_function.out "
            "--error=/path/to/report/%A_%a_my_function.err "
            "--wrap=\"python -c 'from scripts.my_script import my_function; my_function(arg1='value1', arg2=2, batch=\$SLURM_ARRAY_TASK_ID)'\" "
            "--array=1-5 "
            "--dependency=afterok:12345_1,12345_2,12345_3 "
        )
        self.assertEqual(cmd, expected_cmd)


if __name__ == '__main__':
    unittest.main()
