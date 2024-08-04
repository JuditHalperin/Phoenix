import unittest
from tests.interface import Test
from scripts.computation import get_gene_set_batch, get_gene_set_batches


class ComputationTest(Test):

    def setUp(self) -> None:
        self.gene_sets = {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']}
                    
    def test_gene_set_batches(self):
        gene_set_batches = get_gene_set_batches(self.gene_sets, 3)
        self.assertEqual(len(gene_set_batches), 2)
        self.assertEqual([len(batch) for batch in gene_set_batches], [3, 3])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set4', 'set5', 'set6'])

        gene_set_batches = get_gene_set_batches(self.gene_sets, 4)
        self.assertEqual(len(gene_set_batches), 2)
        self.assertEqual([len(batch) for batch in gene_set_batches], [4, 2])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set5', 'set6'])

        gene_set_batches = get_gene_set_batches(self.gene_sets, 2)
        self.assertEqual(len(gene_set_batches), 3)
        self.assertEqual([len(batch) for batch in gene_set_batches], [2, 2, 2])
        self.assertEqual(list(gene_set_batches[-1].keys()), ['set5', 'set6'])

    def test_get_gene_set_batch(self):
        self.assertEqual(get_gene_set_batch(self.gene_sets), self.gene_sets)
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=3), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=2, batch_size=3), {'set4': ['gene4'], 'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=3, batch_size=2), {'set5': ['gene5'], 'set6': ['gene6']})
        self.assertEqual(get_gene_set_batch(self.gene_sets, batch=1, batch_size=4), {'set1': ['gene1'], 'set2': ['gene2'], 'set3': ['gene3'], 'set4': ['gene4']})


if __name__ == '__main__':
    unittest.main()
