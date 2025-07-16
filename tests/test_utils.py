import unittest
import numpy as np
import pandas as pd
from scripts.consts import ALL_CELLS
from tests.interface import Test
from scripts.utils import define_set_size, define_batch_size, convert_to_str, convert_from_str, remove_outliers, correct_effect_size


class UtilTest(Test):

    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
            
    def test_set_size_definition(self):
        self.assertEqual(define_set_size(20, 0.5, 15), 15)
        self.assertEqual(define_set_size(8, 0.5, 10), 8)
        self.assertEqual(define_set_size(80, 0.25, 10), 20)
        self.assertEqual(define_set_size(300, 0.5, 10), 140)  # as 150 is not in SIZES

    def test_num_batches_definition(self):
        self.assertEqual(define_batch_size(9, 3), 3)
        self.assertEqual(define_batch_size(10, 3), 4)
        self.assertEqual(define_batch_size(1, 3), 1)
        
    def test_str_conversion(self):
        self.assertEqual(convert_to_str(2.2), '2.2')
        self.assertEqual(convert_to_str([1, 2]), '1; 2')
        self.assertEqual(convert_to_str({'1': 11, '2': [22, 222]}), '1: 11; 2: 22; 222')

        self.assertEqual(convert_from_str('2.2'), 2.2)
        self.assertEqual(convert_from_str('1; 2'), [1, 2])

    def test_remove_outliers(self):
        data = [11, 12, 12, 13, 12, 100]  # 100 is an outlier
        result = remove_outliers(data)
        assert all(x in result for x in data if x != 100)

        data = [10, 12, 14, 16, 18]
        assert remove_outliers(data) == data

    def test_correct_effect_size_basic(self):
        effect_sizes = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        targets = pd.Series(['A', 'A', 'A', 'B', 'B', ALL_CELLS])
        corrected = correct_effect_size(effect_sizes, targets)

        expected_A = [-1.0, 0.0, 1.0]  # Mean of A group is 2.0
        expected_B = [-5.0, 5.0]      # Mean of B group is 15.0

        np.testing.assert_allclose(corrected[:3], expected_A, rtol=1e-5)
        np.testing.assert_allclose(corrected[3:5], expected_B, rtol=1e-5)
        self.assertEqual(corrected[5], 30.0)  # ALL_CELLS should remain unchanged


if __name__ == '__main__':
    unittest.main()
