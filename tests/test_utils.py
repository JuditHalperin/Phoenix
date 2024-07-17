import unittest
from tests.interface import Test
from scripts.utils import define_set_size, define_batch_size, get_gene_set_batches, convert_to_str, convert_from_str


class UtilTest(Test):
    
    def test_set_size_definition(self):
        self.assertEqual(define_set_size(20, 0.5, 15), 15)
        self.assertEqual(define_set_size(8, 0.5, 10), 8)
        self.assertEqual(define_set_size(80, 0.25, 10), 20)
        self.assertEqual(define_set_size(80, 0.5, 10), 35)  # as 40 is not in SIZES

    def test_num_batches_definition(self):
        self.assertEqual(define_batch_size(9, 3), 3)
        self.assertEqual(define_batch_size(10, 3), 4)
        self.assertEqual(define_batch_size(1, 3), 1)
        
    def test_gene_set_batches(self):
        gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
        gene_set_batches = get_gene_set_batches(gene_sets, 3)
        self.assertEqual(len(gene_set_batches), 2)
        self.assertEqual([len(batch) for batch in gene_set_batches], [3, 3])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set4', 'set5', 'set6'])

        gene_set_batches = get_gene_set_batches(gene_sets, 4)
        self.assertEqual(len(gene_set_batches), 2)
        self.assertEqual([len(batch) for batch in gene_set_batches], [4, 2])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set5', 'set6'])

        gene_set_batches = get_gene_set_batches(gene_sets, 2)
        self.assertEqual(len(gene_set_batches), 3)
        self.assertEqual([len(batch) for batch in gene_set_batches], [2, 2, 2])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set5', 'set6'])

    def test_str_conversion(self):
        self.assertEqual(convert_to_str(2.2), '2.2')
        self.assertEqual(convert_to_str([1, 2]), '1, 2')
        self.assertEqual(convert_to_str({'1': 11, '2': [22, 222]}), '1: 11, 2: 22, 222')

        self.assertEqual(convert_from_str('2.2'), 2.2)
        self.assertEqual(convert_from_str('1, 2'), [1, 2])


if __name__ == '__main__':
    unittest.main()
