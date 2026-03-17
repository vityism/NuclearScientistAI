"""
Data fetcher for collecting atomic/nuclear data from IAEA LiveChart API.
"""
import json
import os
from typing import Dict, List
from datetime import datetime
from config.settings import (
    RAW_DATA_DIR, 
    TRAINING_ELEMENTS, 
    PREDICTION_ELEMENTS,
    NUCLEAR_FEATURES
)
from .iaea_client import IAEAClient


class DataFetcher:
    """Fetches and saves atomic/nuclear data from IAEA LiveChart API."""
    
    def __init__(self):
        """
        Initialize the data fetcher.
        No API key required for IAEA LiveChart API.
        """
        self.client = IAEAClient()
        self.raw_data_dir = RAW_DATA_DIR
        
    def fetch_element_data(self, atomic_number: int) -> Dict:
        """
        Fetch all available nuclear data for an element.
        
        Args:
            atomic_number: The atomic number of the element.
            
        Returns:
            Dictionary containing all nuclear characteristics.
        """
        try:
            data = {
                "atomic_number": atomic_number,
                "timestamp": datetime.now().isoformat(),
                "source": "IAEA LiveChart"
            }
            
            # Get basic element data (includes all isotopes)
            element_data = self.client.get_element_data(atomic_number)
            data.update(element_data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for element {atomic_number}: {str(e)}")
            return {
                "atomic_number": atomic_number,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def fetch_training_data(self) -> List[Dict]:
        """
        Fetch data for all training elements (1-40).
        
        Returns:
            List of dictionaries containing nuclear data.
        """
        print("Fetching training data for elements 1-40 from IAEA LiveChart...")
        training_data = []
        
        for atomic_number in TRAINING_ELEMENTS:
            print(f"  Fetching element {atomic_number}...")
            data = self.fetch_element_data(atomic_number)
            training_data.append(data)
        
        return training_data
    
    def fetch_prediction_targets(self) -> List[int]:
        """
        Get list of elements for prediction (41-118).
        
        Returns:
            List of atomic numbers.
        """
        return PREDICTION_ELEMENTS
    
    def save_raw_data(self, data: List[Dict], filename: str = None) -> str:
        """
        Save raw data to JSON file.
        
        Args:
            data: List of data dictionaries.
            filename: Optional custom filename.
            
        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_nuclear_data_{timestamp}.json"
        
        filepath = os.path.join(self.raw_data_dir, filename)
        
        # Ensure directory exists
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Raw data saved to: {filepath}")
        return filepath
    
    def fetch_and_save_training_data(self) -> str:
        """
        Fetch training data and save to file.
        
        Returns:
            Path to saved file.
        """
        data = self.fetch_training_data()
        return self.save_raw_data(data, "training_data_raw.json")
    
    def load_raw_data(self, filename: str) -> List[Dict]:
        """
        Load raw data from JSON file.
        
        Args:
            filename: Name of the file to load.
            
        Returns:
            List of data dictionaries.
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
