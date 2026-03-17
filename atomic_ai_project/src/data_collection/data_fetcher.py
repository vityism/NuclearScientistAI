"""
Data fetcher for collecting atomic/nuclear data from IAEA API.
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
    """Fetches and saves atomic/nuclear data from IAEA API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            api_key: Optional API key for IAEA API.
        """
        self.client = IAEAClient(api_key=api_key)
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
                "source": "IAEA"
            }
            
            # Get basic element data
            element_data = self.client.get_element_data(atomic_number)
            data.update(element_data)
            
            # Get binding energy
            data["binding_energy"] = self.client.get_binding_energy(atomic_number)
            
            # Get neutron cross-section
            data["neutron_cross_section"] = self.client.get_neutron_cross_section(atomic_number)
            
            # Get isotopic abundance
            data["isotopic_abundance"] = self.client.get_abundance(atomic_number)
            
            # Get data for most stable isotope
            if "most_stable_isotope" in element_data:
                mass_number = element_data["most_stable_isotope"]
                data["half_life"] = self.client.get_half_life(atomic_number, mass_number)
                data["decay_modes"] = self.client.get_decay_modes(atomic_number, mass_number)
                data["spin_parity"] = self.client.get_spin_parity(atomic_number, mass_number)
            
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
        print("Fetching training data for elements 1-40...")
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
