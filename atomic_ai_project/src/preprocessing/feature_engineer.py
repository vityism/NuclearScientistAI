"""
Feature engineering module for atomic/nuclear data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """Creates and transforms features for nuclear property prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features based on atomic number.
        
        Args:
            df: Input DataFrame with cleaned data.
            
        Returns:
            DataFrame with additional derived features.
        """
        df = df.copy()
        
        # Atomic number features
        df['log_atomic_number'] = np.log1p(df['atomic_number'])
        df['atomic_number_squared'] = df['atomic_number'] ** 2
        
        # Neutron-to-proton ratio approximation
        df['n_p_ratio_estimate'] = df['atomic_number'] * 0.5 + 10
        
        # Periodic table position features
        df['period'] = self._get_period(df['atomic_number'])
        df['group'] = self._get_group(df['atomic_number'])
        
        # Block classification (s, p, d, f)
        df['block_s'] = (df['atomic_number'] <= 4).astype(int)
        df['block_p'] = ((df['atomic_number'] > 4) & (df['atomic_number'] <= 18)).astype(int)
        df['block_d'] = ((df['atomic_number'] > 18) & (df['atomic_number'] <= 40)).astype(int)
        df['block_f'] = (df['atomic_number'] > 40).astype(int)
        
        return df
    
    def _get_period(self, atomic_number: pd.Series) -> pd.Series:
        """Determine periodic table period from atomic number."""
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
