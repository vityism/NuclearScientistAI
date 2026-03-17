"""
Data loader for loading and saving processed data.
"""
import json
import os
import pickle
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """Handles loading and saving of processed nuclear data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_raw_json(self, filename: str) -> List[Dict]:
        """
        Load raw data from JSON file.
        
        Args:
            filename: Name of the JSON file.
            
        Returns:
            List of data dictionaries.
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_processed_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed DataFrame to CSV.
        
        Args:
            df: DataFrame to save.
            filename: Output filename.
            
        Returns:
            Path to saved file.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to: {filepath}")
        return filepath
    
    def load_processed_csv(self, filename: str) -> pd.DataFrame:
        """
        Load processed DataFrame from CSV.
        
        Args:
            filename: Name of the CSV file.
            
        Returns:
            Loaded DataFrame.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        return pd.read_csv(filepath)
    
    def save_model_data(self, X: pd.DataFrame, y: pd.DataFrame, 
                        feature_names: List[str], filename: str) -> str:
        """
        Save model training data.
        
        Args:
            X: Feature matrix.
            y: Target values.
            feature_names: List of feature names.
            filename: Output filename.
            
        Returns:
            Path to saved file.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        data = {
            'X': X,
            'y': y,
            'feature_names': feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model data saved to: {filepath}")
        return filepath
    
    def load_model_data(self, filename: str) -> Dict:
        """
        Load model training data.
        
        Args:
            filename: Name of the pickle file.
            
        Returns:
            Dictionary with X, y, and feature_names.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_scaler(self, scaler, filename: str = 'scaler.pkl') -> str:
        """
        Save fitted scaler.
        
        Args:
            scaler: Fitted scaler object.
            filename: Output filename.
            
        Returns:
            Path to saved file.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Scaler saved to: {filepath}")
        return filepath
    
    def load_scaler(self, filename: str = 'scaler.pkl'):
        """
        Load saved scaler.
        
        Args:
            filename: Name of the pickle file.
            
        Returns:
            Loaded scaler object.
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_available_raw_files(self) -> List[str]:
        """
        Get list of available raw data files.
        
        Returns:
            List of filenames.
        """
        if not os.path.exists(self.raw_data_dir):
            return []
        
        return [f for f in os.listdir(self.raw_data_dir) 
                if f.endswith('.json')]
    
    def get_available_processed_files(self) -> List[str]:
        """
        Get list of available processed data files.
        
        Returns:
            List of filenames.
        """
        if not os.path.exists(self.processed_data_dir):
            return []
        
        return [f for f in os.listdir(self.processed_data_dir)]
