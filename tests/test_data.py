import unittest
import pandas as pd
from tests.interface import Test
from scripts.data import intersect_genes, get_top_sum_pathways, get_column_unique_pathways, preprocess


class DataTest(Test):

    def setUp(self):
        self.all_genes = ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Gene6', 'Gene7', 'Gene8']

    def test_preprocessing(self):
        expression = self.generate_data(num_genes=10, mean=5, std=1)
        lowly_expressed = ['Gene3', 'Gene6']
        expression[lowly_expressed] = 1
        expression = preprocess(expression, preprocessed=True, num_genes=8)
        for gene in lowly_expressed:
            assert gene not in expression.columns

    def test_gene_intersection(self):
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene8'], self.all_genes), ['Gene1', 'Gene4', 'Gene8'])
        self.assertEqual(intersect_genes(['GENE1', 'GENE4', 'GENE8'], self.all_genes), ['Gene1', 'Gene4', 'Gene8'])
        self.assertEqual(intersect_genes(['Gene9', 'Gene10'], self.all_genes), [])
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene10'], self.all_genes), ['Gene1', 'Gene4'])
        self.assertEqual(intersect_genes(['gene1', 'gene4', 'gene4'], self.all_genes), ['Gene1', 'Gene4'])

    def test_top_sum_pathways(self):
        df = pd.DataFrame({
            'Target1': [0.1, 0.2, 0.3],
            'Target2': [0.4, None, 0.6],
            'Target3': [0.7, 0.8, 0.9]
        }, index=['Pathway1', 'Pathway2', 'Pathway3'])

        result = get_top_sum_pathways(df, ascending=False, size=2)
        expected = ['Pathway3', 'Pathway1']
        self.assertEqual(result, expected)

        result = get_top_sum_pathways(df, ascending=True, size=2)
        expected = ['Pathway1', 'Pathway3']
        self.assertEqual(result, expected)

    def test_column_unique_pathways(self):
        """
        Pathway5 is above threshold, Pathway6 is not the minimum
        """
        df = pd.DataFrame({
            'Target1': [0.1, 0.2, 0.3, 0.4, 0.6, 0.2],
            'Target2': [0.4, None, 0.6, 0.7, 0.8, 0.1],
            'Target3': [0.7, 0.8, 0.9, 0.5, 0.6, 0.2],
            'Target4': [0.2, 0.3, 0.6, 0.6, 0.7, 0.2]
        }, index=['Pathway1', 'Pathway2', 'Pathway3', 'Pathway4', 'Pathway5', 'Pathway6'])

        result = get_column_unique_pathways(df, 'Target1', size=10, threshold=0.5)
        expected = ['Pathway3', 'Pathway1', 'Pathway2', 'Pathway4']
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
