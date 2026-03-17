"""
Model trainer for nuclear energy level prediction.
Trains on energy levels of isotopes from elements 1-40 and predicts for 41-118.
"""
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from config.settings import MODEL_DIR, TRAINING_ELEMENTS, PREDICTION_ELEMENTS, ENERGY_LEVEL_CONFIG
from .nuclear_predictor import EnergyLevelPredictor


class EnergyLevelTrainer:
    """Handles the complete training pipeline for energy level prediction."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.model_dir = MODEL_DIR
        self.model = None
        self.feature_names = []
        self.max_levels = ENERGY_LEVEL_CONFIG["max_levels_per_isotope"]
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     feature_names: List[str],
                     validation_split: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix [num_isotopes, input_dim].
            y: Target matrix [num_isotopes, max_levels, 2].
            feature_names: Names of features.
            validation_split: Fraction of data for validation.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        self.feature_names = feature_names
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]} isotopes")
        print(f"Validation set size: {X_val.shape[0]} isotopes")
        print(f"Energy levels per isotope: {y_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    input_dim: int,
                    feature_names: List[str],
                    epochs: int = None,
                    batch_size: int = None) -> EnergyLevelPredictor:
        """
        Train the energy level prediction model.
        
        Args:
            X_train: Training features [num_isotopes, input_dim].
            y_train: Training targets [num_isotopes, max_levels, 2].
            X_val: Validation features.
            y_val: Validation targets.
            input_dim: Number of input features.
            feature_names: List of feature names.
            epochs: Number of training epochs.
            batch_size: Batch size.
            
        Returns:
            Trained EnergyLevelPredictor model.
        """
        self.feature_names = feature_names
        
        # Initialize model
        self.model = EnergyLevelPredictor(
            input_dim=input_dim,
            max_levels=self.max_levels
        )
        
        # Build and train
        self.model.build()
        self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate trained model on test data.
        
        Args:
            X_test: Test features.
            y_test: Test targets [num_isotopes, max_levels, 2].
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        metrics = self.model.evaluate(X_test, y_test)
        
        print("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        
        return metrics
    
    def predict_energy_levels(self, X: np.ndarray, 
                              isotope_info: List[Dict]) -> List[Dict]:
        """
        Predict energy levels for specific isotopes.
        
        Args:
            X: Feature matrix [num_isotopes, input_dim].
            isotope_info: List of dicts with 'atomic_number' and 'mass_number'.
            
        Returns:
            List of dictionaries with predicted energy levels for each isotope.
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        return self.model.predict_energy_levels_table(X, isotope_info)
    
    def predict_elements_range(self, X: np.ndarray,
                               atomic_numbers: List[int]) -> Dict:
        """
        Make predictions for a range of elements (41-118).
        
        Args:
            X: Feature matrix.
            atomic_numbers: List of atomic numbers to predict.
            
        Returns:
            Dictionary with predictions organized by element and isotope.
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        predictions = self.model.predict(X)
        
        results = {}
        for i, atomic_num in enumerate(atomic_numbers):
            if atomic_num not in results:
                results[atomic_num] = {
                    'element_symbol': self._get_element_symbol(atomic_num),
                    'isotopes': []
                }
            
            # Each row in X should correspond to an isotope
            results[atomic_num]['isotopes'].append({
                'energy_levels_raw': predictions[i],
                'max_levels': self.max_levels
            })
        
        return results
    
    def _get_element_symbol(self, atomic_number: int) -> str:
        """Get element symbol from atomic number."""
        elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        if 1 <= atomic_number <= len(elements):
            return elements[atomic_number - 1]
        return f"Z{atomic_number}"
    
    def save_model(self, filename: str = 'energy_level_predictor.h5') -> str:
        """
        Save trained model to file.
        
        Args:
            filename: Output filename.
            
        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = os.path.join(self.model_dir, filename)
        return self.model.save(filepath)
    
    def load_model(self, filename: str = 'energy_level_predictor.h5') -> EnergyLevelPredictor:
        """
        Load trained model from file.
        
        Args:
            filename: Name of saved model file.
            
        Returns:
            Loaded EnergyLevelPredictor model.
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model = EnergyLevelPredictor(input_dim=1)
        self.model.load(filepath)
        
        return self.model
    
    def train_and_save(self, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str],
                       validation_split: float = 0.2,
                       epochs: int = None,
                       batch_size: int = None,
                       model_filename: str = 'energy_level_predictor.h5') -> Dict:
        """
        Complete training pipeline: prepare data, train, evaluate, and save.
        
        Args:
            X: Feature matrix [num_isotopes, input_dim].
            y: Target matrix [num_isotopes, max_levels, 2].
            feature_names: Names of features.
            validation_split: Fraction for validation.
            epochs: Number of training epochs.
            batch_size: Batch size.
            model_filename: Output model filename.
            
        Returns:
            Dictionary with training results and metrics.
        """
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(
            X, y, feature_names, validation_split
        )
        
        # Train model
        self.train_model(
            X_train, y_train,
            X_val, y_val,
            input_dim=X.shape[1],
            feature_names=feature_names,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate on validation set
        val_metrics = self.evaluate_model(X_val, y_val)
        
        # Save model
        model_path = self.save_model(model_filename)
        
        # Save metadata
        metadata = {
            'feature_names': feature_names,
            'input_dim': X.shape[1],
            'output_shape': y.shape[1:],  # (max_levels, 2)
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'validation_metrics': val_metrics,
            'model_path': model_path
        }
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_path}")
        
        return metadata
