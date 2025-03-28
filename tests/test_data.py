import unittest
import pandas as pd
import numpy as np
from tests.interface import Test
from scripts.data import preprocess_expression, preprocess_data, scale_expression, scale_pseudotime
from scripts.consts import CELL_TYPE_COL


class PreprocessingTest(Test):

    def setUp(self) -> None:
        self.expression = self.generate_data(3, 3)

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['Type1', 'Type2', 'Type1']
        }, index=self.expression.index)

        self.pseudotime = pd.DataFrame({
            'Lineage1': [0.1, 0.2, np.nan],
            'Lineage2': [0.4, 0.5, 0.6]
        }, index=self.expression.index)

        self.reduction = pd.DataFrame({
            'UMAP1': [0.1, 0.2, 0.3],
            'UMAP2': [0.4, 0.5, 0.6]
        }, index=self.expression.index)

    def test_preprocessing_flow(self):
        preprocess_data(self.expression, self.cell_types, self.pseudotime, self.reduction, verbose=False)

    def test_gene_filtering(self):
        expression = self.generate_data(num_genes=10, mean=5, std=1)
        lowly_expressed = ['Gene3', 'Gene6']
        expression[lowly_expressed] = 1
        expression = preprocess_expression(expression, preprocessed=True, num_genes=8, verbose=False)
        for gene in lowly_expressed:
            assert gene not in expression.columns

    def test_scaled_expression(self):
        expression = pd.DataFrame({
            'Gene1': [1, 2, 3],
            'Gene2': [4, 5, 6],
            'Gene3': [7, 8, 9]
        }, index=['Cell1', 'Cell2', 'Cell3'])
        scaled_expression = scale_expression(expression)
        assert all(scaled_expression.index == expression.index) and all(scaled_expression.columns == expression.columns)
        assert all(scaled_expression.mean() == 0) and all(round(scaled_expression.std(ddof=0), 7) == 1)
        assert scaled_expression.iloc[1, 1] == (5 - np.mean([4, 5, 6])) / np.std([4, 5, 6])
        assert scaled_expression.iloc[1, 2] == (8 - np.mean([7, 8, 9])) / np.std([7, 8, 9])

    def test_scaled_pseudotime(self):
        pseudotime = pd.DataFrame({
            'Lineage1': [0.1, 0.2, np.nan],
            'Lineage2': [0.4, 0.5, 0.6]
        }, index=['Cell1', 'Cell2', 'Cell3'])
        scaled_pseudotime = scale_pseudotime(pseudotime)
        assert all(scaled_pseudotime.index == pseudotime.index) and all(scaled_pseudotime.columns == pseudotime.columns)
        assert all(scaled_pseudotime.min() == 0) and all(scaled_pseudotime.max() == 1)
        assert scaled_pseudotime.iloc[1, 1] == (0.5 - np.min([0.4, 0.5, 0.6])) / (np.max([0.4, 0.5, 0.6]) - np.min([0.4, 0.5, 0.6]))
        assert scaled_pseudotime.iloc[1, 0] == (0.2 - np.min([0.1, 0.2])) / (np.max([0.1, 0.2]) - np.min([0.1, 0.2]))


if __name__ == '__main__':
    unittest.main()
