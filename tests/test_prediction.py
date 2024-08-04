import unittest
import pandas as pd
import numpy as np
from tests.interface import Test
from scripts.prediction import get_target, get_data, train, compare_scores
from scripts.utils import adjust_p_value
from scripts.consts import CELL_TYPE_COL, ALL_CELLS, CLASSIFIERS, CLASSIFIER_ARGS, REGRESSORS, REGRESSOR_ARGS, THRESHOLD


class TrainDataTest(Test):

    def setUp(self):

        self.expression = pd.DataFrame({
            'Gene1': [1.0, 1.3, 1.5, 1.7],  # similar across cell types
            'Gene2': [10, 1, 10, 1],  # very different across cell types
            'Gene3': [3, 3, 3, 5],  # a little different across cell types
            'Gene4': [4, 4, 4, 4],  # identical across cell types
            'Gene5': [10, 1, 10, 1],  # very different across cell types
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        self.cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        self.pseudotime = pd.DataFrame({
            1: [0.1, 0.2, 0.3, None],
            2: [0.6, 0.3, 0.9, 0.1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

    def test_cell_type_target(self):
        # Test case for get_target function with cell_types
        cell_type = 'TypeB'
        cell_type_target = get_target(cell_types=self.cell_types, cell_type=cell_type)
        self.assertTrue(cell_type_target.dtype == bool)
        self.assertEqual(cell_type_target.tolist(), [c == cell_type for c in self.cell_types[CELL_TYPE_COL]])

        cell_type_target = get_target(cell_types=self.cell_types, cell_type=ALL_CELLS)
        self.assertEqual(cell_type_target.tolist(), self.cell_types[CELL_TYPE_COL].tolist())

    def test_pseudotime_target(self):
        # Test case for get_target function with pseudotime
        pseudotime_target = get_target(pseudotime=self.pseudotime, lineage=2, scale=False)
        self.assertEqual(pseudotime_target.tolist(), self.pseudotime[2].tolist())

        pseudotime_target = get_target(pseudotime=self.pseudotime, lineage=1, scale=True)
        self.assertEqual(pseudotime_target.tolist(), [0.0, 0.5, 1.0])

    def test_data_without_selection(self):
        # Test case where using cell_types and cell_type
        X, _, features = get_data(
            self.expression,
            features=self.expression.columns, 
            cell_types=self.cell_types,
            cell_type='TypeA',
            set_size=None,
            feature_selection=None
        )

        self.assertEqual(X.shape, self.expression.shape)
        self.assertListEqual(features, self.expression.columns.tolist())

        X, _, features = get_data(
            self.expression,
            features=self.expression.columns, 
            cell_types=self.cell_types,
            cell_type='TypeA',
            set_size=self.expression.shape[1],  # all
            feature_selection='RF'
        )

        self.assertEqual(X.shape, self.expression.shape)
        self.assertListEqual(features, self.expression.columns.tolist())

    def test_cell_type_data_with_anova(self):
        # Test case where using cell_types and cell_type
        cell_dim = self.expression.shape[0]

        for cell_type in ['All', 'TypeA', 'TypeB']:

            set_size = 4
            X, y, selected_genes = get_data(
                self.expression,
                features=self.expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertTrue('Gene4' not in selected_genes)

            set_size = 3
            X, y, selected_genes = get_data(
                self.expression,
                features=['Gene1', 'Gene3', 'Gene2', 'Gene4', 'Gene5'], 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(selected_genes, ['Gene3', 'Gene2', 'Gene5']) # SelectKBest does not change order of features

            set_size = 2
            X, y, selected_genes = get_data(
                self.expression,
                features=self.expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(selected_genes, ['Gene2', 'Gene5']) # SelectKBest does not change order of features

            X, y, selected_genes = get_data(
                self.expression,
                features=['Gene1', 'Gene3', 'Gene5'],  # missing good predictor Gene2
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='ANOVA'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)
            self.assertEqual(selected_genes, ['Gene3', 'Gene5']) # SelectKBest does not change order of features
            
    def test_cell_type_data_with_rf(self):
        cell_dim = self.expression.shape[0]

        for cell_type in ['All', 'TypeA', 'TypeB']:

            set_size = self.expression.shape[1] - 1
            X, y, selected_genes = get_data(
                self.expression,
                features=self.expression.columns, 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='RF'
            )

            self.assertEqual(X.shape, (cell_dim, set_size))
            self.assertEqual(len(y), cell_dim)
            self.assertEqual(len(selected_genes), set_size)

            # RandomForestClassifier orders features by importance:
            self.assertTrue(selected_genes[0] in ['Gene5', 'Gene2']) 
            self.assertTrue(selected_genes[1] in ['Gene5', 'Gene2'])
            self.assertTrue('Gene4' not in selected_genes) 

            set_size = self.expression.shape[1] - 2
            X, y, selected_genes = get_data(
                self.expression,
                features=[g for g in self.expression.columns if g != 'Gene2'], 
                cell_types=self.cell_types,
                cell_type=cell_type,
                set_size=set_size,
                feature_selection='RF'
            )

            self.assertEqual(len(selected_genes), set_size)
            self.assertTrue(selected_genes[0] == 'Gene5') 
            self.assertTrue('Gene4' not in selected_genes) 

    def test_pseudotime_data_with_anova(self):
        # Test case where using pseudotime and lineage

        lineage = 1
        cell_dim = self.pseudotime[lineage].dropna().shape[0]
        set_size = 1
        X, y, selected_genes = get_data(
            expression=self.expression,
            features=self.expression.columns,
            pseudotime=self.pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='ANOVA'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertTrue(selected_genes[0] == 'Gene1')  # Gene1 increases across time in lineage 1

        lineage = 2
        cell_dim = self.pseudotime[lineage].dropna().shape[0]
        set_size = 3
        X, y, selected_genes = get_data(
            expression=self.expression,
            features=self.expression.columns,
            pseudotime=self.pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='ANOVA'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(selected_genes, ['Gene2', 'Gene3', 'Gene5'])  # these 3 genes increase / decrease according to pseudotime order (and SelectKBest does not change order of features)

    def test_pseudotime_data_with_rf(self):
        # Test case where using pseudotime and lineage

        set_size = self.expression.shape[1] - 1

        lineage = 1
        cell_dim = self.pseudotime[lineage].dropna().shape[0]
        X, y, selected_genes = get_data(
            expression=self.expression,
            features=self.expression.columns,
            pseudotime=self.pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='RF'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertTrue(selected_genes[0] == 'Gene1')  # Gene1 increases across time in lineage 1

        lineage = 2
        cell_dim = self.pseudotime[lineage].dropna().shape[0]
        X, y, selected_genes = get_data(
            expression=self.expression,
            features=self.expression.columns,
            pseudotime=self.pseudotime,
            lineage=lineage,
            set_size=set_size,
            feature_selection='RF'
        )
        
        self.assertEqual(X.shape, (cell_dim, set_size))
        self.assertEqual(len(y), cell_dim)
        self.assertEqual(len(selected_genes), set_size)
        self.assertEqual(selected_genes[0], 'Gene1')  # changes across each pseudotime step
        self.assertTrue('Gene4' not in selected_genes)  # does not change across pseudotime order


class TrainingTest(Test):
    
    def setUp(self):

        self.expression = pd.DataFrame({
            'Gene1': [1.0, 1.3, 1.5, 1.7],  # similar across cell types
            'Gene2': [10, 1, 10, 1],  # very different across cell types
            'Gene3': [3, 3, 3, 5],  # a little different across cell types
            'Gene4': [4, 4, 4, 4],  # identical across cell types
            'Gene5': [10, 1, 10, 1],  # very different across cell types
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        cell_types = pd.DataFrame({
            CELL_TYPE_COL: ['TypeA', 'TypeB', 'TypeA', 'TypeB'],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.cell_type_target = get_target(cell_types=cell_types, cell_type=ALL_CELLS)

        pseudotime = pd.DataFrame({
            1: [0.1, 0.2, 0.3, 0.4],
            2: [0.6, 0.3, 0.9, 0.1],
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        self.pseudotime1_target = get_target(pseudotime=pseudotime, lineage=1)
        self.pseudotime2_target = get_target(pseudotime=pseudotime, lineage=2)

        self.cross_validation = 2
    
    def test_classification_training(self):

        good_features = np.array(self.expression[['Gene2', 'Gene5']])
        middle_features = np.array(self.expression[['Gene2', 'Gene1']])
        bad_features = np.array(self.expression[['Gene4', 'Gene1']])
        
        for classifier in ['RF', 'SVM', 'DTree']:
            for metric in ['f1_weighted', 'balanced_accuracy']:

                predictor = CLASSIFIERS[classifier]
                predictor_args = CLASSIFIER_ARGS[predictor]
                
                good_score = train(good_features, self.cell_type_target, predictor, predictor_args, metric, self.cross_validation)
                middle_score = train(middle_features, self.cell_type_target, predictor, predictor_args, metric, self.cross_validation)
                bad_score = train(bad_features, self.cell_type_target, predictor, predictor_args, metric, self.cross_validation)

                self.assertEqual(good_score, 1)
                self.assertGreaterEqual(good_score, middle_score)
                self.assertGreaterEqual(middle_score, bad_score)

    def test_regression_training(self):

        # Lineage 1
        good_features = np.array(self.expression[['Gene1']])
        good_features2 = np.array(self.expression[['Gene3']])
        bad_features = np.array(self.expression[['Gene4']])
        
        for classifier in ['RF', 'SVM', 'LGBM']:
            for metric in ['neg_mean_squared_error']:

                predictor = REGRESSORS[classifier]
                predictor_args = REGRESSOR_ARGS[predictor]
                
                good_score = train(good_features, self.pseudotime1_target, predictor, predictor_args, metric, self.cross_validation)
                good_score2 = train(good_features2, self.pseudotime1_target, predictor, predictor_args, metric, self.cross_validation)
                bad_score = train(bad_features, self.pseudotime1_target, predictor, predictor_args, metric, self.cross_validation)

                self.assertGreaterEqual(good_score, bad_score)
                self.assertGreaterEqual(good_score2, bad_score)

        # Lineage 2
        good_features = np.array(self.expression[['Gene2']])
        good_features2 = np.array(self.expression[['Gene5']])
        bad_features = np.array(self.expression[['Gene4']])
        
        for classifier in ['RF', 'SVM', 'LGBM']:
            for metric in ['neg_mean_squared_error']:

                predictor = REGRESSORS[classifier]
                predictor_args = REGRESSOR_ARGS[predictor]
                
                good_score = train(good_features, self.pseudotime2_target, predictor, predictor_args, metric, self.cross_validation)
                good_score2 = train(good_features2, self.pseudotime2_target, predictor, predictor_args, metric, self.cross_validation)
                bad_score = train(bad_features, self.pseudotime2_target, predictor, predictor_args, metric, self.cross_validation)

                self.assertGreaterEqual(good_score, bad_score)
                self.assertGreaterEqual(good_score2, bad_score)


class ScoreComparisonTest(Test):
    
    def test_score_comparison(self):
        background_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert compare_scores(pathway_score=0.7, background_scores=background_scores, distribution='normal') <= THRESHOLD
        assert compare_scores(pathway_score=0.2, background_scores=background_scores, distribution='normal') > THRESHOLD

    def test_multiple_corrections(self):
        adjust_p_value([0.0, 0.05, 0.5, 1.0])


if __name__ == '__main__':
    unittest.main()
