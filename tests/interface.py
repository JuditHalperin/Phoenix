import unittest
import pandas as pd
import numpy as np


class Test(unittest.TestCase):

    def generate_data(
            self,
            num_genes: int = 100,
            num_cells: int = 50,
            mean: int | float = 5,
            std: int | float = 1,
    ) -> pd.DataFrame:
        
        data = np.random.lognormal(mean, std, size=(num_cells, num_genes))
        data = np.log1p(data)

        return pd.DataFrame(
            data,
            columns=[f'Gene{i}' for i in range(num_genes)],
            index=[f'Cell{i}' for i in range(num_cells)]
        )
