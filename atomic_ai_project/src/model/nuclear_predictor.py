"""
Neural network model for predicting nuclear energy levels.
Predicts excitation energies for each isotope individually.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from config.settings import HYPERPARAMETERS, ENERGY_LEVEL_CONFIG


class EnergyLevelPredictor:
    """
    Neural network model for predicting energy levels of isotopes.
    Outputs a matrix/table of energy levels for each isotope.
    """
    
    def __init__(self, input_dim: int, 
                 max_levels: int = None,
                 hyperparams: Optional[Dict] = None):
        """
        Initialize the energy level predictor model.
        
        Args:
            input_dim: Number of input features (Z, A, and context features).
            max_levels: Maximum number of energy levels to predict per isotope.
            hyperparams: Optional dictionary of hyperparameters.
        """
        self.input_dim = input_dim
        self.max_levels = max_levels or ENERGY_LEVEL_CONFIG["max_levels_per_isotope"]
        self.hyperparams = hyperparams or HYPERPARAMETERS
        self.model = None
        self.history = None
        
        # Energy level configuration
        self.min_energy = ENERGY_LEVEL_CONFIG["min_energy"]
        self.max_energy = ENERGY_LEVEL_CONFIG["max_energy"]
        self.energy_unit = ENERGY_LEVEL_CONFIG["energy_unit"]
    
    def _build_model(self) -> Model:
        """
        Build the neural network architecture for sequence prediction.
        Uses LSTM/GRU for sequential energy level prediction.
        
        Returns:
            Compiled Keras Model.
        """
        # Input: [Z, A, previous_levels...]
        inputs = keras.Input(shape=(self.input_dim,))
        
        x = inputs
        
        # Dense layers for feature processing
        for units in self.hyperparams['hidden_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.hyperparams['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
        # Reshape for sequence output
        # Output shape: (max_levels, 2) where 2 = [energy, spin_parity_encoded]
        x = layers.RepeatVector(self.max_levels)(x)
        
        # LSTM layers for sequence generation
        x = layers.LSTM(128, return_sequences=True, activation='relu')(x)
        x = layers.Dropout(self.hyperparams['dropout_rate'])(x)
        x = layers.LSTM(64, return_sequences=True, activation='relu')(x)
        x = layers.Dropout(self.hyperparams['dropout_rate'])(x)
        
        # Output layer: [energy_level, spin_parity] for each level
        outputs = layers.TimeDistributed(
            layers.Dense(2, activation='linear')
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='energy_level_predictor')
        
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
            X_train: Training features [Z, A, ...].
            y_train: Training targets [num_isotopes, max_levels, 2].
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
        Predict energy levels for isotopes.
        
        Args:
            X: Input features [num_isotopes, input_dim].
            
        Returns:
            Predicted energy levels [num_isotopes, max_levels, 2].
            Where [:, :, 0] = energy in keV, [:, :, 1] = spin_parity encoded.
        """
        if self.model is None:
            raise ValueError("Model must be built or loaded before prediction")
        
        predictions = self.model.predict(X)
        
        # Ensure energy values are within physical bounds
        predictions[:, :, 0] = np.clip(predictions[:, :, 0], 
                                        self.min_energy, self.max_energy)
        
        return predictions
    
    def predict_energy_levels_table(self, X: np.ndarray, 
                                    isotope_info: List[Dict]) -> List[Dict]:
        """
        Predict energy levels and format as a readable table/matrix.
        
        Args:
            X: Input features.
            isotope_info: List of dicts with 'atomic_number' and 'mass_number'.
            
        Returns:
            List of dictionaries, one per isotope, containing:
                - atomic_number
                - mass_number
                - energy_levels: list of {level, energy_keV, spin_parity}
        """
        predictions = self.predict(X)
        
        results = []
        for i, info in enumerate(isotope_info):
            isotope_result = {
                "atomic_number": info["atomic_number"],
                "mass_number": info["mass_number"],
                "element_symbol": self._get_element_symbol(info["atomic_number"]),
                "energy_levels": []
            }
            
            for level_idx in range(self.max_levels):
                energy = predictions[i, level_idx, 0]
                spin_parity_encoded = predictions[i, level_idx, 1]
                
                # Skip levels with near-zero energy after ground state
                if level_idx > 0 and energy < 1.0:  # Less than 1 keV
                    continue
                
                # Decode spin-parity from encoded value
                spin_parity = self._decode_spin_parity(spin_parity_encoded)
                
                isotope_result["energy_levels"].append({
                    "level": level_idx,
                    "energy_keV": float(energy),
                    "spin_parity": spin_parity
                })
            
            results.append(isotope_result)
        
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
    
    def _decode_spin_parity(self, encoded_value: float) -> str:
        """
        Decode spin-parity from encoded numerical value.
        This is a simplified mapping; real implementation would use proper encoding.
        """
        # Common spin-parity values and their encodings
        spin_parities = [
            (0.0, "0+"), (0.5, "1/2+"), (1.0, "1+"), (1.5, "3/2+"),
            (2.0, "2+"), (2.5, "5/2+"), (3.0, "3+"), (3.5, "7/2+"),
            (4.0, "4+"), (4.5, "9/2+"), (5.0, "5+"),
            (0.1, "0-"), (0.6, "1/2-"), (1.1, "1-"), (1.6, "3/2-"),
            (2.1, "2-"), (2.6, "5/2-"), (3.1, "3-"), (3.6, "7/2-"),
            (4.1, "4-"), (4.6, "9/2-"), (5.1, "5-")
        ]
        
        # Find closest match
        min_diff = float('inf')
        closest = "unknown"
        for encoded, label in spin_parities:
            diff = abs(encoded_value - encoded)
            if diff < min_diff:
                min_diff = diff
                closest = label
        
        return closest
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features.
            y: True targets [num_isotopes, max_levels, 2].
            
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
        print(f"Model loaded from: {filepath}")
        return self.model
    
    def export_predictions_to_csv(self, predictions: List[Dict], 
                                  filepath: str) -> str:
        """
        Export predicted energy levels to CSV file (matrix/table format).
        
        Args:
            predictions: List of prediction dictionaries from predict_energy_levels_table.
            filepath: Path to save CSV.
            
        Returns:
            Path to saved CSV file.
        """
        import csv
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['atomic_number', 'element', 'mass_number', 
                         'level', 'energy_keV', 'spin_parity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for isotope in predictions:
                for level in isotope['energy_levels']:
                    row = {
                        'atomic_number': isotope['atomic_number'],
                        'element': isotope['element_symbol'],
                        'mass_number': isotope['mass_number'],
                        'level': level['level'],
                        'energy_keV': round(level['energy_keV'], 3),
                        'spin_parity': level['spin_parity']
                    }
                    writer.writerow(row)
        
        print(f"Predictions exported to: {filepath}")
        return filepath
