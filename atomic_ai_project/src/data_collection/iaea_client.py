"""
IAEA API Client for fetching atomic and nuclear data.
"""
import requests
import json
from typing import Dict, List, Optional
from config.settings import IAEA_API_BASE_URL, IAEA_API_KEY


class IAEAClient:
    """Client for interacting with the IAEA atomic/nuclear database API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the IAEA API client.
        
        Args:
            api_key: API key for authentication. If None, uses environment variable.
        """
        self.api_key = api_key or IAEA_API_KEY
        self.base_url = IAEA_API_BASE_URL
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def get_element_data(self, atomic_number: int) -> Dict:
        """
        Fetch atomic/nuclear data for a specific element.
        
        Args:
            atomic_number: The atomic number of the element (1-118).
            
        Returns:
            Dictionary containing nuclear characteristics.
            
        Raises:
            requests.RequestException: If API request fails.
        """
        endpoint = f"{self.base_url}/nuclides/{atomic_number}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    def get_nuclide_data(self, atomic_number: int, mass_number: int) -> Dict:
        """
        Fetch data for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            Dictionary containing nuclide-specific data.
        """
        endpoint = f"{self.base_url}/nuclides/{atomic_number}/{mass_number}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    def get_binding_energy(self, atomic_number: int) -> float:
        """
        Get binding energy for an element.
        
        Args:
            atomic_number: The atomic number.
            
        Returns:
            Binding energy in MeV.
        """
        data = self.get_element_data(atomic_number)
        return data.get("binding_energy", 0.0)
    
    def get_half_life(self, atomic_number: int, mass_number: int) -> str:
        """
        Get half-life for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            Half-life string (e.g., "stable", "1.23e+9 years").
        """
        data = self.get_nuclide_data(atomic_number, mass_number)
        return data.get("half_life", "unknown")
    
    def get_decay_modes(self, atomic_number: int, mass_number: int) -> List[str]:
        """
        Get decay modes for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            List of decay mode strings.
        """
        data = self.get_nuclide_data(atomic_number, mass_number)
        return data.get("decay_modes", [])
    
    def get_spin_parity(self, atomic_number: int, mass_number: int) -> str:
        """
        Get spin and parity for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            Spin-parity string (e.g., "0+", "7/2-").
        """
        data = self.get_nuclide_data(atomic_number, mass_number)
        return data.get("spin_parity", "unknown")
    
    def get_neutron_cross_section(self, atomic_number: int) -> float:
        """
        Get neutron cross-section for an element.
        
        Args:
            atomic_number: The atomic number.
            
        Returns:
            Neutron cross-section in barns.
        """
        data = self.get_element_data(atomic_number)
        return data.get("neutron_cross_section", 0.0)
    
    def get_abundance(self, atomic_number: int) -> Dict[str, float]:
        """
        Get natural abundance for isotopes of an element.
        
        Args:
            atomic_number: The atomic number.
            
        Returns:
            Dictionary mapping mass numbers to abundances (percentage).
        """
        data = self.get_element_data(atomic_number)
        return data.get("isotopic_abundance", {})
