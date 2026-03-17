"""
Neural network model for predicting nuclear properties.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from config.settings import HYPERPARAMETERS


class NuclearPredictor:
    """Neural network model for predicting atomic/nuclear characteristics."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hyperparams: Optional[Dict] = None):
        """
        Initialize the nuclear predictor model.
        
        Args:
            input_dim: Number of input features.
            output_dim: Number of output targets.
            hyperparams: Optional dictionary of hyperparameters.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyperparams = hyperparams or HYPERPARAMETERS
        self.model = None
        self.history = None
    
    def _build_model(self) -> Model:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras Model.
        """
        inputs = keras.Input(shape=(self.input_dim,))
        
        x = inputs
        
        # Hidden layers with dropout
        for units in self.hyperparams['hidden_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.hyperparams['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.output_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='nuclear_predictor')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.hyperparams['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build(self) -> Model:
        """
        Build and return the model.
        
        Returns:
            Compiled Keras Model.
        """
        self.model = self._build_model()
        self.model.summary()
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            
        Returns:
            Training history.
        """
        if self.model is None:
            self.build()
        
        epochs = epochs or self.hyperparams['epochs']
        batch_size = batch_size or self.hyperparams['batch_size']
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model must be built or loaded before prediction")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features.
            y: True targets.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be built or loaded before evaluation")
        
        metrics = self.model.evaluate(X, y, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, metrics))
    
    def save(self, filepath: str) -> str:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model.
            
        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load(self, filepath: str) -> Model:
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model.
            
        Returns:
            Loaded Keras Model.
        """
        self.model = keras.models.load_model(filepath)
        self.input_dim = self.model.input_shape[-1]
        self.output_dim = self.model.output_shape[-1]
        print(f"Model loaded from: {filepath}")
        return self.model
    
    def get_feature_importance(self, X: np.ndarray, method: str = 'permutation') -> np.ndarray:
        """
        Estimate feature importance.
        
        Args:
            X: Input features.
            method: Importance calculation method.
            
        Returns:
            Array of feature importance scores.
        """
        if self.model is None:
            raise ValueError("Model must be loaded for feature importance")
        
        if method == 'permutation':
            # Simple permutation importance
            baseline_pred = self.model.predict(X)
            baseline_mse = np.mean((baseline_pred - self.model.predict(X)) ** 2)
            
            importance = []
            for i in range(X.shape[1]):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_pred = self.model.predict(X_permuted)
                permuted_mse = np.mean((permuted_pred - baseline_pred) ** 2)
                importance.append(permuted_mse - baseline_mse)
            
            return np.array(importance)
        
        else:
            raise ValueError(f"Unknown method: {method}")
