"""
Model trainer for nuclear property prediction.
"""
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from config.settings import MODEL_DIR, TRAINING_ELEMENTS, PREDICTION_ELEMENTS
from .nuclear_predictor import NuclearPredictor


class ModelTrainer:
    """Handles the complete training pipeline for nuclear prediction."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.model_dir = MODEL_DIR
        self.model = None
        self.feature_names = []
        self.target_names = []
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     target_names: List[str],
                     validation_split: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix.
            y: Target matrix.
            target_names: Names of target variables.
            validation_split: Fraction of data for validation.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        self.target_names = target_names
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    input_dim: int, output_dim: int,
                    feature_names: List[str],
                    epochs: int = None,
                    batch_size: int = None) -> NuclearPredictor:
        """
        Train the nuclear prediction model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            input_dim: Number of input features.
            output_dim: Number of output targets.
            feature_names: List of feature names.
            epochs: Number of training epochs.
            batch_size: Batch size.
            
        Returns:
            Trained NuclearPredictor model.
        """
        self.feature_names = feature_names
        
        # Initialize model
        self.model = NuclearPredictor(
            input_dim=input_dim,
            output_dim=output_dim
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
            y_test: Test targets.
            
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
    
    def predict_elements(self, X: np.ndarray, atomic_numbers: List[int]) -> Dict:
        """
        Make predictions for specific elements.
        
        Args:
            X: Feature matrix.
            atomic_numbers: List of atomic numbers.
            
        Returns:
            Dictionary with predictions organized by element.
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        predictions = self.model.predict(X)
        
        results = {}
        for i, atomic_num in enumerate(atomic_numbers):
            results[atomic_num] = {
                'predictions': predictions[i],
                'target_names': self.target_names
            }
        
        return results
    
    def save_model(self, filename: str = 'nuclear_predictor.h5') -> str:
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
    
    def load_model(self, filename: str = 'nuclear_predictor.h5') -> NuclearPredictor:
        """
        Load trained model from file.
        
        Args:
            filename: Name of saved model file.
            
        Returns:
            Loaded NuclearPredictor model.
        """
        filepath = os.path.join(self.model_dir, filename)
        self.model = NuclearPredictor(input_dim=1, output_dim=1)
        self.model.load(filepath)
        
        return self.model
    
    def train_and_save(self, X: np.ndarray, y: np.ndarray,
                       target_names: List[str],
                       feature_names: List[str],
                       validation_split: float = 0.2,
                       epochs: int = None,
                       batch_size: int = None,
                       model_filename: str = 'nuclear_predictor.h5') -> Dict:
        """
        Complete training pipeline: prepare data, train, evaluate, and save.
        
        Args:
            X: Feature matrix.
            y: Target matrix.
            target_names: Names of target variables.
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
            X, y, target_names, validation_split
        )
        
        # Train model
        self.train_model(
            X_train, y_train,
            X_val, y_val,
            input_dim=X.shape[1],
            output_dim=y.shape[1],
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
            'target_names': target_names,
            'input_dim': X.shape[1],
            'output_dim': y.shape[1],
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'validation_metrics': val_metrics,
            'model_path': model_path
        }
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_path}")
        
        return metadata
