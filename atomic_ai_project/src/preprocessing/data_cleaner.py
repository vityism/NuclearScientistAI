"""
Data cleaning module for atomic/nuclear data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from config.settings import NUCLEAR_FEATURES


class DataCleaner:
    """Cleans and validates atomic/nuclear data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.required_features = NUCLEAR_FEATURES
    
    def clean_half_life(self, half_life_str: str) -> Optional[float]:
        """
        Convert half-life string to seconds.
        
        Args:
            half_life_str: Half-life string (e.g., "stable", "1.23e+9 years").
            
        Returns:
            Half-life in seconds, or None if conversion fails.
        """
        if not half_life_str or half_life_str.lower() == "stable":
            return np.inf
        
        # Parse time value and unit
        try:
            parts = half_life_str.split()
            if len(parts) < 2:
                return None
            
            value = float(parts[0].replace('e+', 'e').replace('E+', 'E'))
            unit = parts[1].lower()
            
            # Convert to seconds
            conversion_factors = {
                'seconds': 1,
                'second': 1,
                's': 1,
                'minutes': 60,
                'minute': 60,
                'min': 60,
                'hours': 3600,
                'hour': 3600,
                'h': 3600,
                'days': 86400,
                'day': 86400,
                'd': 86400,
                'years': 3.154e+7,
                'year': 3.154e+7,
                'y': 3.154e+7,
                'ms': 1e-3,
                'μs': 1e-6,
                'us': 1e-6,
                'ns': 1e-9,
            }
            
            return value * conversion_factors.get(unit, 1)
            
        except (ValueError, IndexError):
            return None
    
    def clean_decay_modes(self, decay_modes: List[str]) -> Dict[str, float]:
        """
        Convert decay modes to numerical features.
        
        Args:
            decay_modes: List of decay mode strings.
            
        Returns:
            Dictionary with decay mode probabilities.
        """
        decay_mode_map = {
            'alpha': 0,
            'beta-': 1,
            'beta+': 2,
            'electron_capture': 3,
            'gamma': 4,
            'neutron_emission': 5,
            'proton_emission': 6,
            'spontaneous_fission': 7,
            'stable': 8
        }
        
        if not decay_modes:
            return {'dominant_mode': 8, 'mode_count': 0}
        
        # Encode dominant decay mode
        dominant = decay_modes[0].lower().replace('-', '_').replace('+', '_plus')
        dominant_code = decay_mode_map.get(dominant, -1)
        
        return {
            'dominant_mode': dominant_code if dominant_code >= 0 else -1,
            'mode_count': len(decay_modes)
        }
    
    def clean_spin_parity(self, spin_parity: str) -> Dict[str, float]:
        """
        Parse spin-parity string into numerical values.
        
        Args:
            spin_parity: Spin-parity string (e.g., "0+", "7/2-").
            
        Returns:
            Dictionary with spin and parity values.
        """
        if not spin_parity or spin_parity == "unknown":
            return {'spin': np.nan, 'parity': np.nan}
        
        try:
            # Extract spin value
            if '/' in spin_parity:
                parts = spin_parity.split('/')
                spin = float(parts[0]) / float(parts[1].rstrip('+-'))
            else:
                spin = float(spin_parity.rstrip('+-'))
            
            # Extract parity
            parity = 0
            if '+' in spin_parity:
                parity = 1
            elif '-' in spin_parity:
                parity = -1
            
            return {'spin': spin, 'parity': parity}
            
        except (ValueError, IndexError):
            return {'spin': np.nan, 'parity': np.nan}
    
    def clean_isotope_data(self, isotope_data: Dict, atomic_number: int) -> Dict:
        """
        Clean data for a single isotope.
        
        Args:
            isotope_data: Raw isotope data dictionary.
            atomic_number: Atomic number of the element.
            
        Returns:
            Cleaned isotope data dictionary with atomic_number and mass_number.
        """
        mass_number = isotope_data.get('mass_number', 0)
        
        cleaned = {
            'atomic_number': atomic_number,
            'mass_number': mass_number
        }
        
        # Clean binding energy
        cleaned['binding_energy'] = isotope_data.get('binding_energy', np.nan)
        
        # Clean half-life
        half_life_raw = isotope_data.get('half_life', 'unknown')
        cleaned['half_life_seconds'] = self.clean_half_life(half_life_raw)
        
        # Clean decay modes
        decay_modes_raw = isotope_data.get('decay_modes', [])
        decay_info = self.clean_decay_modes(decay_modes_raw)
        cleaned.update(decay_info)
        
        # Clean spin-parity
        spin_parity_raw = isotope_data.get('spin_parity', 'unknown')
        spin_info = self.clean_spin_parity(spin_parity_raw)
        cleaned.update(spin_info)
        
        # Clean neutron cross-section
        cleaned['neutron_cross_section'] = isotope_data.get('neutron_cross_section', np.nan)
        
        # Store energy levels directly
        cleaned['energy_levels'] = isotope_data.get('energy_levels', [])
        
        return cleaned
    
    def clean_element_data(self, element_data: Dict) -> List[Dict]:
        """
        Clean all data for a single element, returning one record per isotope.
        
        Args:
            element_data: Raw element data dictionary with 'atomic_number' and 'isotopes' list.
            
        Returns:
            List of cleaned isotope data dictionaries (one per isotope).
        """
        atomic_number = element_data.get('atomic_number', 0)
        isotopes = element_data.get('isotopes', [])
        
        if not isotopes:
            # If no isotopes found, return a single placeholder record
            return [{
                'atomic_number': atomic_number,
                'mass_number': atomic_number,  # Use Z as placeholder for A
                'binding_energy': np.nan,
                'half_life_seconds': np.nan,
                'dominant_mode': np.nan,
                'mode_count': 0,
                'spin': np.nan,
                'parity': np.nan,
                'neutron_cross_section': np.nan,
                'abundance': np.nan,
                'isotope_count': 0,
                'energy_levels': []
            }]
        
        # Clean each isotope separately
        cleaned_isotopes = []
        for isotope in isotopes:
            cleaned_isotope = self.clean_isotope_data(isotope, atomic_number)
            cleaned_isotopes.append(cleaned_isotope)
        
        return cleaned_isotopes
    
    def clean_dataset(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Clean entire dataset.
        
        Args:
            raw_data: List of raw element data dictionaries (each containing 'atomic_number' and 'isotopes').
            
        Returns:
            Pandas DataFrame with cleaned data (one row per isotope).
        """
        all_cleaned_isotopes = []
        
        for element_data in raw_data:
            # clean_element_data now returns a list of isotope records
            cleaned_isotopes = self.clean_element_data(element_data)
            all_cleaned_isotopes.extend(cleaned_isotopes)
        
        df = pd.DataFrame(all_cleaned_isotopes)
        
        # Sort by atomic number, then mass number
        df = df.sort_values(['atomic_number', 'mass_number']).reset_index(drop=True)
        
        return df
