"""
Unit tests for the nuclear prediction AI project.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataCleaner(unittest.TestCase):
    """Tests for DataCleaner class."""
    
    def setUp(self):
        from src.preprocessing.data_cleaner import DataCleaner
        self.cleaner = DataCleaner()
    
    def test_clean_half_life_stable(self):
        """Test cleaning stable half-life."""
        result = self.cleaner.clean_half_life("stable")
        self.assertEqual(result, np.inf)
    
    def test_clean_half_life_years(self):
        """Test cleaning half-life in years."""
        result = self.cleaner.clean_half_life("1.23e+9 years")
        expected = 1.23e+9 * 3.154e+7
        self.assertAlmostEqual(result, expected, places=-6)
    
    def test_clean_spin_parity(self):
        """Test cleaning spin-parity string."""
        result = self.cleaner.clean_spin_parity("7/2-")
        self.assertEqual(result['spin'], 3.5)
        self.assertEqual(result['parity'], -1)
    
    def test_clean_spin_parity_positive(self):
        """Test cleaning positive parity."""
        result = self.cleaner.clean_spin_parity("0+")
        self.assertEqual(result['spin'], 0.0)
        self.assertEqual(result['parity'], 1)


class TestFeatureEngineer(unittest.TestCase):
    """Tests for FeatureEngineer class."""
    
    def setUp(self):
        from src.preprocessing.feature_engineer import FeatureEngineer
        self.engineer = FeatureEngineer()
    
    def test_add_derived_features(self):
        """Test adding derived features."""
        df = pd.DataFrame({'atomic_number': [1, 6, 26]})
        result = self.engineer.add_derived_features(df)
        
        self.assertIn('log_atomic_number', result.columns)
        self.assertIn('period', result.columns)
        self.assertIn('group', result.columns)
    
    def test_get_period(self):
        """Test period assignment."""
        atomic_numbers = pd.Series([1, 6, 11, 19, 37, 55])
        periods = self.engineer._get_period(atomic_numbers)
        
        self.assertEqual(periods.iloc[0], 1)  # H
        self.assertEqual(periods.iloc[1], 2)  # C
        self.assertEqual(periods.iloc[2], 3)  # Na
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.inf, 2.0, 3.0]
        })
        result = self.engineer.handle_missing_values(df, strategy='median')
        
        self.assertFalse(result['a'].isna().any())
        self.assertFalse(np.isinf(result['b']).any())


class TestNuclearPredictor(unittest.TestCase):
    """Tests for NuclearPredictor model."""
    
    def setUp(self):
        from src.model.nuclear_predictor import NuclearPredictor
        self.model = NuclearPredictor(input_dim=10, output_dim=3)
    
    def test_build_model(self):
        """Test building the model."""
        model = self.model.build()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[-1], 10)
        self.assertEqual(model.output_shape[-1], 3)
    
    def test_predict_shape(self):
        """Test prediction output shape."""
        self.model.build()
        
        X_test = np.random.randn(5, 10)
        predictions = self.model.predict(X_test)
        
        self.assertEqual(predictions.shape, (5, 3))


class TestEvaluationMetrics(unittest.TestCase):
    """Tests for EvaluationMetrics class."""
    
    def setUp(self):
        from src.evaluation.metrics import EvaluationMetrics
        self.evaluator = EvaluationMetrics()
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [2.9, 4.2], [5.2, 5.8]])
        
        metrics = self.evaluator.calculate_all_metrics(
            y_true, y_pred, 
            target_names=['target1', 'target2']
        )
        
        self.assertIn('overall', metrics)
        self.assertIn('per_target', metrics)
        self.assertIn('mse', metrics['overall'])
        self.assertIn('r2', metrics['overall'])
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = y_true.copy()
        
        metrics = self.evaluator.calculate_all_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['overall']['r2'], 1.0)
        self.assertAlmostEqual(metrics['overall']['mse'], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_data_flow(self):
        """Test data flows through preprocessing correctly."""
        from src.preprocessing.data_cleaner import DataCleaner
        from src.preprocessing.feature_engineer import FeatureEngineer
        
        # Sample raw data
        raw_data = [{
            'atomic_number': 6,
            'binding_energy': 7.68,
            'half_life': 'stable',
            'neutron_cross_section': 0.0035,
            'isotopic_abundance': {12: 0.989, 13: 0.011}
        }]
        
        # Clean data
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataset(raw_data)
        
        # Engineer features
        engineer = FeatureEngineer()
        target_cols = ['binding_energy', 'half_life_seconds']
        X, y, feature_names = engineer.prepare_features(
            cleaned_df, target_cols=target_cols, fit=True
        )
        
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(len(feature_names), 0)


if __name__ == '__main__':
    unittest.main()
