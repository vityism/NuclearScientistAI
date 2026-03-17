"""
IAEA LiveChart API Client for fetching atomic and nuclear data.
This client uses the IAEA LiveChart API which does not require an API key.
Focus: Energy levels (excitation energies) for each isotope.
"""
import requests
import json
from typing import Dict, List, Optional, Any
from config.settings import IAEA_LIVECHART_BASE_URL


class IAEAClient:
    """Client for interacting with the IAEA LiveChart nuclear database API."""
    
    def __init__(self):
        """
        Initialize the IAEA LiveChart API client.
        No API key is required for this public API.
        """
        self.base_url = IAEA_LIVECHART_BASE_URL
        self.session = requests.Session()
        # Set user agent to be respectful
        self.session.headers.update({
            "User-Agent": "Atomic-AI-Project/1.0",
            "Accept": "application/json"
        })
    
    def _fetch_nuclide_data(self, atomic_number: int, mass_number: int) -> Dict:
        """
        Fetch raw data for a specific nuclide from IAEA LiveChart.
        
        Args:
            atomic_number: The atomic number (Z).
            mass_number: The mass number (A).
            
        Returns:
            Dictionary containing nuclide data.
        """
        url = self.base_url.format(atomic_number, mass_number)
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_element_data(self, atomic_number: int) -> Dict:
        """
        Fetch atomic/nuclear data for all isotopes of a specific element.
        Includes energy level data for each isotope.
        
        Args:
            atomic_number: The atomic number of the element (1-118).
            
        Returns:
            Dictionary containing nuclear characteristics for all isotopes,
            including energy levels for each isotope.
            
        Raises:
            requests.RequestException: If API request fails.
        """
        element_data = {
            "atomic_number": atomic_number,
            "isotopes": []
        }
        
        # Common range of mass numbers for stability estimation
        min_mass = atomic_number  # At least Z protons
        max_mass = atomic_number + 50  # Reasonable upper bound
        
        for mass_num in range(min_mass, min(max_mass, min_mass + 20)):
            try:
                nuclide_data = self._fetch_nuclide_data(atomic_number, mass_num)
                if nuclide_data:
                    # Extract energy levels for this isotope
                    nuclide_data["energy_levels"] = self.get_energy_levels(atomic_number, mass_num)
                    element_data["isotopes"].append(nuclide_data)
            except requests.RequestException:
                continue
                
        return element_data
    
    def get_nuclide_data(self, atomic_number: int, mass_number: int) -> Dict:
        """
        Fetch data for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            Dictionary containing nuclide-specific data including energy levels.
        """
        data = self._fetch_nuclide_data(atomic_number, mass_number)
        data["energy_levels"] = self.get_energy_levels(atomic_number, mass_number)
        return data
    
    def get_energy_levels(self, atomic_number: int, mass_number: int) -> List[Dict[str, Any]]:
        """
        Get energy levels (excitation energies) for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            List of dictionaries, each containing:
                - energy: Excitation energy in keV
                - spin_parity: Jπ value
                - half_life: Half-life at this energy level (if applicable)
                - decay_mode: Decay mode from this level (if applicable)
        """
        # Fetch from IAEA LiveChart - energy level endpoint
        # Note: Actual implementation depends on IAEA API structure
        # This is a placeholder that would be adapted to actual API response
        try:
            url = f"{self.base_url}&levels=true".format(atomic_number, mass_number)
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("energy_levels", [])
        except Exception:
            pass
        
        # Fallback: return empty list if API doesn't provide detailed levels
        return []
    
    def get_binding_energy(self, atomic_number: int, mass_number: int) -> float:
        """
        Get binding energy for a specific nuclide.
        
        Args:
            atomic_number: The atomic number.
            mass_number: The mass number.
            
        Returns:
            Binding energy in MeV.
        """
        data = self.get_nuclide_data(atomic_number, mass_number)
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
