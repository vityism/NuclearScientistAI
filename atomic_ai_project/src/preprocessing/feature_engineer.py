"""
Feature engineering module for atomic/nuclear energy level data.
Creates features and prepares energy level sequences for training.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """Creates and transforms features for nuclear energy level prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False
    
    def prepare_input_features(self, df: pd.DataFrame, 
                               feature_cols: List[str],
                               fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare input features from DataFrame.
        
        Args:
            df: Input DataFrame with cleaned data.
            feature_cols: List of column names to use as features.
            fit: Whether to fit the scaler (True for training, False for prediction).
            
        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        # Ensure required columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            # Create minimal features from atomic_number and mass_number
            X = df[['atomic_number', 'mass_number']].values
            feature_names = ['atomic_number', 'mass_number']
        else:
            X = df[available_cols].values
            feature_names = available_cols
        
        # Scale features
        X_scaled, _ = self.scale_features(df, feature_names, fit=fit)
        
        return X_scaled, feature_names
    
    def prepare_prediction_features(self, prediction_isotopes: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for prediction isotopes.
        
        Args:
            prediction_isotopes: List of dicts with 'atomic_number' and 'mass_number'.
            
        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        # Create feature matrix with same structure as training data
        X_pred = np.array([[iso['atomic_number'], iso['mass_number']] 
                           for iso in prediction_isotopes])
        feature_names = ['atomic_number', 'mass_number']
        
        return X_pred, feature_names
    
    def prepare_energy_level_targets(self, df: pd.DataFrame,
                                     max_levels: int = 50) -> Tuple[np.ndarray, List[Dict]]:
        """
        Prepare energy level sequences as targets for training.
        
        Args:
            df: Input DataFrame with energy level data.
            max_levels: Maximum number of energy levels per isotope.
            
        Returns:
            Tuple of (target_array, isotope_info).
            target_array shape: [num_isotopes, max_levels, 2]
                where [:, :, 0] = energy in keV, [:, :, 1] = spin_parity_encoded
        """
        num_isotopes = len(df)
        targets = np.zeros((num_isotopes, max_levels, 2))
        isotope_info = []
        
        for idx, row in df.iterrows():
            atomic_num = row.get('atomic_number', 0)
            mass_num = row.get('mass_number', 0)
            
            isotope_info.append({
                'atomic_number': int(atomic_num),
                'mass_number': int(mass_num)
            })
            
            # Get energy levels if available
            energy_levels = row.get('energy_levels', [])
            
            if isinstance(energy_levels, list) and len(energy_levels) > 0:
                for level_idx, level_data in enumerate(energy_levels[:max_levels]):
                    # Extract energy value
                    if isinstance(level_data, dict):
                        energy = level_data.get('energy', 0.0)
                        spin_parity = level_data.get('spin_parity', '0+')
                    else:
                        energy = float(level_data) if len(level_data) > 0 else 0.0
                        spin_parity = '0+'
                    
                    # Store energy
                    targets[idx, level_idx, 0] = float(energy)
                    
                    # Encode spin-parity
                    targets[idx, level_idx, 1] = self._encode_spin_parity(spin_parity)
            else:
                # Set ground state (level 0) to 0 energy
                targets[idx, 0, 0] = 0.0
                targets[idx, 0, 1] = self._encode_spin_parity('0+')
        
        return targets, isotope_info
    
    def _encode_spin_parity(self, spin_parity: str) -> float:
        """
        Encode spin-parity string to numerical value.
        
        Args:
            spin_parity: String like "0+", "7/2-", "1/2+", etc.
            
        Returns:
            Encoded float value.
        """
        # Mapping of common spin-parity values
        encoding_map = {
            "0+": 0.0, "1/2+": 0.5, "1+": 1.0, "3/2+": 1.5,
            "2+": 2.0, "5/2+": 2.5, "3+": 3.0, "7/2+": 3.5,
            "4+": 4.0, "9/2+": 4.5, "5+": 5.0,
            "0-": 0.1, "1/2-": 0.6, "1-": 1.1, "3/2-": 1.6,
            "2-": 2.1, "5/2-": 2.6, "3-": 3.1, "7/2-": 3.6,
            "4-": 4.1, "9/2-": 4.6, "5-": 5.1
        }
        
        if spin_parity in encoding_map:
            return encoding_map[spin_parity]
        
        # Try to parse fractional spins
        try:
            if '/' in spin_parity:
                parts = spin_parity.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1].replace('+', '').replace('-', ''))
                    base_value = numerator / denominator
                    # Add parity offset
                    if '-' in spin_parity:
                        base_value += 0.1
                    return base_value
        except:
            pass
        
        # Default to 0+
        return 0.0
    
    def scale_features(self, df: pd.DataFrame, 
                      feature_names: List[str],
                      fit: bool = True) -> Tuple[np.ndarray, StandardScaler]:
        """
        Scale features using StandardScaler.
        
        Args:
            df: Input DataFrame.
            feature_names: List of feature column names.
            fit: Whether to fit the scaler or just transform.
            
        Returns:
            Tuple of (scaled_features, scaler).
        """
        available_cols = [col for col in feature_names if col in df.columns]
        
        if not available_cols:
            X = df[['atomic_number', 'mass_number']].values
        else:
            X = df[available_cols].values
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted before transforming")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, self.scaler
    
    def scale_prediction_features(self, X_raw: np.ndarray, 
                                  feature_names: List[str]) -> np.ndarray:
        """
        Scale raw prediction features using fitted scaler.
        
        Args:
            X_raw: Raw feature matrix [num_samples, num_features].
            feature_names: Names of features (for validation).
            
        Returns:
            Scaled feature matrix.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before scaling prediction data")
        
        # Ensure X_raw has the correct number of features
        if X_raw.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"X_raw has {X_raw.shape[1]} features, but scaler expects {self.scaler.n_features_in_} features"
            )
        
        return self.scaler.transform(X_raw)
    
    def _get_period(self, atomic_number: pd.Series) -> pd.Series:
        """Get period number from atomic number."""
        periods = pd.Series(index=atomic_number.index, dtype=int)
        
        periods[atomic_number <= 2] = 1
        periods[(atomic_number > 2) & (atomic_number <= 10)] = 2
        periods[(atomic_number > 10) & (atomic_number <= 18)] = 3
        periods[(atomic_number > 18) & (atomic_number <= 36)] = 4
        periods[(atomic_number > 36) & (atomic_number <= 54)] = 5
        periods[(atomic_number > 54) & (atomic_number <= 86)] = 6
        periods[atomic_number > 86] = 7
        
        return periods
    
    def _get_group(self, atomic_number: pd.Series) -> pd.Series:
        """Approximate group number from atomic number."""
        # Simplified group assignment
        groups = pd.Series(index=atomic_number.index, dtype=int)
        groups[:] = (atomic_number % 18).replace(0, 18)
        return groups
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame.
            strategy: Strategy for handling missing values ('mean', 'median', 'ffill').
            
        Returns:
            DataFrame with handled missing values.
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'ffill':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Replace infinite values with median
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                median_val = df[col][np.isfinite(df[col])].median()
                df[col] = df[col].replace([np.inf, -np.inf], median_val)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str], 
                       fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame.
            feature_cols: List of columns to scale.
            fit: Whether to fit the scaler (True for training, False for inference).
            
        Returns:
            Tuple of (DataFrame with scaled features, scaled array).
        """
        df = df.copy()
        
        X = df[feature_cols].values
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted before transforming")
            X_scaled = self.scaler.transform(X)
        
        # Update DataFrame with scaled values
        for i, col in enumerate(feature_cols):
            df[col] = X_scaled[:, i]
        
        return df, X_scaled
    
    def prepare_features(self, df: pd.DataFrame, target_cols: List[str],
                         fit: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for model training.
        
        Args:
            df: Input DataFrame.
            target_cols: List of target column names.
            fit: Whether to fit transformers.
            
        Returns:
            Tuple of (X, y, feature_names).
        """
        # Add derived features
        df = self.add_derived_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Identify feature columns (exclude targets and non-features)
        exclude_cols = ['atomic_number'] + target_cols
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in [np.float64, np.int64, float, int]]
        
        # Scale features
        df, X = self.scale_features(df, feature_cols, fit=fit)
        
        # Extract targets
        y = df[target_cols].values
        
        return X, y, feature_cols
    
    def create_prediction_features(self, atomic_numbers: List[int]) -> pd.DataFrame:
        """
        Create feature DataFrame for elements to predict.
        
        Args:
            atomic_numbers: List of atomic numbers for prediction.
            
        Returns:
            DataFrame with features for prediction.
        """
        df = pd.DataFrame({'atomic_number': atomic_numbers})
        
        # Add placeholder values for other features
        df['binding_energy'] = np.nan
        df['half_life_seconds'] = np.nan
        df['dominant_mode'] = np.nan
        df['mode_count'] = 0
        df['spin'] = np.nan
        df['parity'] = np.nan
        df['neutron_cross_section'] = np.nan
        df['abundance'] = np.nan
        df['isotope_count'] = 0
        
        # Add derived features
        df = self.add_derived_features(df)
        
        return df
