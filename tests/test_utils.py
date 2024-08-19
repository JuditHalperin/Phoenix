import unittest
from tests.interface import Test
from scripts.utils import define_set_size, define_batch_size, convert_to_str, convert_from_str


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


if __name__ == '__main__':
    unittest.main()
