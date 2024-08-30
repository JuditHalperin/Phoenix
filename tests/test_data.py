import unittest
import pandas as pd
import numpy as np
from tests.interface import Test
from scripts.data import preprocess_expression, preprocess_data
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
        preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            min_lineage_percent=0, min_cell_percent=0, last_cells=0, last_survived_cells=0, verbose=False
        )

    def test_cell_type_exclusion(self):
        expression, cell_types, _, _ = preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            preprocessed=True, exclude_cell_types=['Type1'],
            min_cell_percent=0, min_lineage_percent=0, last_cells=0, last_survived_cells=0, verbose=False
        )
        self.assertEqual(expression.shape[0], 1)
        self.assertEqual(cell_types[CELL_TYPE_COL].tolist(), ['Type2'])

    def test_rare_cell_types(self):
        expression, cell_types, _, _ = preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            preprocessed=True, min_cell_percent=50,  # only Type1 is above 50 percent
            min_lineage_percent=0, last_cells=0, last_survived_cells=0, verbose=False
        )
        self.assertEqual(expression.shape[0], 2)
        self.assertEqual(cell_types[CELL_TYPE_COL].tolist(), ['Type1', 'Type1'])

    def test_lineage_exclusion(self):
        _, _, pseudotime, _ = preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            preprocessed=True, exclude_lineages=['Lineage1'],
            min_cell_percent=0, min_lineage_percent=0, last_cells=0, last_survived_cells=0, verbose=False
        )
        self.assertEqual(pseudotime.columns.tolist(), ['Lineage2'])

    def test_short_lineages(self):
        _, _, pseudotime, _ = preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            preprocessed=True, min_lineage_percent=10,  # Lineage1 has NA value
            min_cell_percent=0, last_cells=0, last_survived_cells=0, verbose=False
        )
        self.assertEqual(pseudotime.columns.tolist(), ['Lineage2'])

    def test_last_cells_in_lineage(self):
        _, _, pseudotime, _ = preprocess_data(
            self.expression, self.cell_types, self.pseudotime, self.reduction,
            preprocessed=True, exclude_cell_types=['Type2'], last_cells=1, last_survived_cells=1,
            min_cell_percent=0, min_lineage_percent=0, verbose=False
        )
        self.assertEqual(pseudotime.columns.tolist(), ['Lineage2'])

    def test_gene_filtering(self):
        expression = self.generate_data(num_genes=10, mean=5, std=1)
        lowly_expressed = ['Gene3', 'Gene6']
        expression[lowly_expressed] = 1
        expression = preprocess_expression(expression, preprocessed=True, num_genes=8, verbose=False)
        for gene in lowly_expressed:
            assert gene not in expression.columns


if __name__ == '__main__':
    unittest.main()
