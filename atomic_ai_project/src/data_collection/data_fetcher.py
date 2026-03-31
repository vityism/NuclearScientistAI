"""
Data fetcher for collecting atomic/nuclear data from IAEA LiveChart API.
"""
import json
import os
from typing import Dict, List
from datetime import datetime
from config.settings import (
    RAW_DATA_DIR, 
    NUCLEAR_FEATURES
)
from .iaea_client import IAEAClient


class DataFetcher:
    """Fetches and saves atomic/nuclear data from IAEA LiveChart API."""
    
    def __init__(self, training_elements=None):
        """
        Initialize the data fetcher.
        
        Args:
            training_elements: Optional list of atomic numbers to use for training.
                              If None, uses default range (1-40).
        """
        self.client = IAEAClient()
        self.raw_data_dir = RAW_DATA_DIR
        
        # Use provided training elements or fall back to default
        if training_elements is not None:
            self.training_elements = training_elements
        else:
            # Import defaults from settings
            from config.settings import DEFAULT_TRAINING_START, DEFAULT_TRAINING_END
            self.training_elements = list(range(DEFAULT_TRAINING_START, DEFAULT_TRAINING_END + 1))
        
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
        Fetch data for all training elements.
        
        Returns:
            List of dictionaries containing nuclear data.
        """
        start_elem = min(self.training_elements)
        end_elem = max(self.training_elements)
        print(f"Fetching training data for elements {start_elem}-{end_elem} from IAEA LiveChart...")
        training_data = []
        
        for atomic_number in self.training_elements:
            print(f"  Fetching element {atomic_number}...")
            data = self.fetch_element_data(atomic_number)
            training_data.append(data)
        
        return training_data
    
    def fetch_prediction_targets(self, training_end: int = 40) -> List[int]:
        """
        Get list of elements for prediction (training_end+1 to 118).
        
        Args:
            training_end: The last atomic number used for training.
            
        Returns:
            List of atomic numbers for prediction.
        """
        return list(range(training_end + 1, 119))
    
    def fetch_valid_prediction_isotopes(self, start_atomic: int, end_atomic: int) -> List[Dict]:
        """
        Fetch all valid isotopes from IAEA database for prediction elements.
        This method queries the database to ensure we only predict for real isotopes.
        
        Args:
            start_atomic: Starting atomic number (inclusive).
            end_atomic: Ending atomic number (inclusive).
            
        Returns:
            List of dictionaries with 'atomic_number' and 'mass_number' for all valid isotopes.
        """
        # Use the IAEA client to get all valid isotopes
        return self.client.get_all_prediction_isotopes(start_atomic, end_atomic)
    
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
