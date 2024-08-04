import unittest
import pandas as pd
from tests.interface import Test
from run_batch import run_task
from scripts.consts import CELL_TYPE_COL, ALL_CELLS, CLASSIFICATION_METRIC, REGRESSION_METRIC, FEATURE_SELECTION, SEED, THRESHOLD


class TaskRunTest(Test):

    def setUp(self) -> None:

        self.expression = pd.DataFrame({
            'Gene1': [1, 3, 5, 7, 9, 11],
            'Gene2': [10, 1, 10, 1, 10, 1],
            'Gene3': [3, 4, 4, 5, 6, 7],
            'Gene4': [4, 4, 4, 4, 4, 4],
            'Gene5': [10, 1, 10, 1, 10, 1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.pseudotime = pd.DataFrame({
            1: [0.1, 0.25, 0.3, None, 0.44, 0.5],
            2: [0.6, 0.3, 0.9, 0.1, 1.0, 1.2],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'])

        self.cross_validation = 3
        self.predictors = ['RF', 'Reg', 'DTree']
    
    def test_classification_run(self):

        for predictor in self.predictors:
            task_args = {
                'expression': self.expression,
                'predictor': predictor,
                'metric': CLASSIFICATION_METRIC,
                'set_size': 1,
                'feature_selection': FEATURE_SELECTION,
                'cross_validation': self.cross_validation,
                'repeats': 50,
                'seed': SEED,
                'distribution': 'gamma',
                'cell_types': self.cell_types,
                'cell_type': ALL_CELLS,
                'cache': None  # avoid saving to cache during test
            }
            
            good_gene_set = ['Gene2']
            p_value = run_task(gene_set=good_gene_set, **task_args)[0]
            self.assertLessEqual(p_value, THRESHOLD)
            good_gene_set = ['Gene5']
            p_value = run_task(gene_set=good_gene_set, **task_args)[0]
            self.assertLessEqual(p_value, THRESHOLD)

            bad_gene_set = ['Gene4']
            p_value = run_task(gene_set=bad_gene_set, **task_args)[0]
            self.assertGreaterEqual(p_value, THRESHOLD)
            bad_gene_set = ['Gene1']
            p_value = run_task(gene_set=bad_gene_set, **task_args)[0]
            self.assertGreaterEqual(p_value, THRESHOLD)

    def test_regression_run(self):

        for predictor in self.predictors:
            task_args = {
                'expression': self.expression,
                'predictor': predictor,
                'metric': REGRESSION_METRIC,
                'set_size': 1,
                'feature_selection': FEATURE_SELECTION,
                'cross_validation': self.cross_validation,
                'repeats': 10,
                'seed': SEED,
                'distribution': 'gamma',
                'pseudotime': self.pseudotime,
                'lineage': 1,
                'cache': None  # avoid saving to cache during test
            }
            
            good_gene_set = ['Gene1']
            p_value = run_task(gene_set=good_gene_set, **task_args)[0]
            self.assertLessEqual(p_value, THRESHOLD)
            good_gene_set = ['Gene3']
            p_value = run_task(gene_set=good_gene_set, **task_args)[0]
            self.assertLessEqual(p_value, THRESHOLD)

            bad_gene_set = ['Gene2']
            p_value = run_task(gene_set=bad_gene_set, **task_args)[0]
            self.assertGreaterEqual(p_value, THRESHOLD)
            bad_gene_set = ['Gene4']
            p_value = run_task(gene_set=bad_gene_set, **task_args)[0]
            self.assertGreaterEqual(p_value, THRESHOLD)
        

if __name__ == '__main__':
    unittest.main()
