"""
IAEA LiveChart API Client for fetching atomic and nuclear data.
This client uses the IAEA LiveChart API which does not require an API key.
Focus: Energy levels (excitation energies) for each isotope.

API Documentation: https://www-nds.iaea.org/relnsd/vcharthtml/api_v0_guide.html
Note: The API returns CSV format, not JSON.
"""
import requests
import csv
import io
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class IAEAClient:
    """Client for interacting with the IAEA LiveChart nuclear database API."""
    
    # Official API Base URL as per documentation
    BASE_URL = "https://nds.iaea.org/relnsd/v1/data"
    
    def __init__(self):
        """
        Initialize the IAEA LiveChart API client.
        No API key is required for this public API.
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Atomic-AI-Project/1.0",
            "Accept": "text/csv"
        })
    
    def get_valid_isotopes(self, atomic_number: int) -> List[int]:
        """
        Get list of valid mass numbers (isotopes) for a specific element
        by querying the IAEA API directly.
        
        Args:
            atomic_number: The atomic number of the element (1-118).
            
        Returns:
            List of valid mass numbers for this element from IAEA database.
        """
        symbol = self._get_symbol(atomic_number)
        if not symbol:
            logger.error(f"Unknown symbol for Z={atomic_number}")
            return []
        
        # Fetch all nuclide data at once and filter locally
        # Using 'fields=levels' endpoint which returns full CSV with Z,A columns
        try:
            params = {
                "nuclides": "all",
                "fields": "levels"
            }
            response = self.session.get(self.BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            
            # Parse CSV to extract unique mass numbers for this Z
            valid_masses = set()
            lines = response.text.strip().split('\n')
            
            # Skip header line
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        z_val = int(parts[0])
                        a_val = int(parts[1])
                        if z_val == atomic_number:
                            valid_masses.add(a_val)
                    except (ValueError, IndexError):
                        continue
            
            return sorted(list(valid_masses))
            
        except Exception as e:
            logger.error(f"Error fetching isotopes for Z={atomic_number}: {e}")
            return []
    
    def get_energy_levels(self, atomic_number: int, mass_number: int) -> List[Dict]:
        """
        Fetch energy level data for a specific isotope from IAEA API.
        
        Args:
            atomic_number: Atomic number (Z).
            mass_number: Mass number (A).
            
        Returns:
            List of dictionaries containing energy level data.
        """
        symbol = self._get_symbol(atomic_number)
        if not symbol:
            logger.error(f"Unknown symbol for Z={atomic_number}")
            return []
        
        nuclide_id = f"{mass_number}{symbol}".lower()
        params = {
            "nuclides": nuclide_id,
            "fields": "levels"
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # API returns CSV format
            return self._parse_levels_csv(response.text)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"HTTP 404: Isotope {nuclide_id} not found")
            else:
                logger.error(f"HTTP Error for {nuclide_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching levels for {nuclide_id}: {e}")
            return []
    
    def _parse_levels_csv(self, csv_text: str) -> List[Dict]:
        """Parse CSV response from IAEA API into list of level dictionaries."""
        levels = []
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                level_data = {}
                
                # Extract energy (in keV)
                if 'energy' in row and row['energy']:
                    try:
                        level_data['energy_keV'] = float(row['energy'])
                    except ValueError:
                        continue
                
                # Extract spin-parity
                if 'jp' in row and row['jp']:
                    level_data['spin_parity'] = row['jp']
                
                # Extract half-life if present
                if 'half_life_sec' in row and row['half_life_sec']:
                    try:
                        level_data['half_life_sec'] = float(row['half_life_sec'])
                    except ValueError:
                        pass
                
                if 'energy_keV' in level_data:
                    levels.append(level_data)
                    
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
        
        return levels
    
    def _get_symbol(self, z: int) -> Optional[str]:
        """Returns the chemical symbol for a given atomic number."""
        symbols = [
            "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
            "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
            "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        ]
        if 1 <= z < len(symbols):
            return symbols[z]
        return None
    
    def verify_prediction_isotopes(self, prediction_isotopes: List[Tuple[int, int]], 
                                   start_atomic: int, end_atomic: int) -> Dict:
        """
        Double-checks that predictions match exactly the valid isotopes in the database.
        
        Args:
            prediction_isotopes: List of (Z, A) tuples that were predicted.
            start_atomic: Start of prediction range (inclusive).
            end_atomic: End of prediction range (inclusive).
            
        Returns:
            Dictionary with verification results.
        """
        valid_isotopes_set = set()
        
        logger.info("Verifying predictions against IAEA database...")
        for z in range(start_atomic, end_atomic + 1):
            valid_masses = self.get_valid_isotopes(z)
            for a in valid_masses:
                valid_isotopes_set.add((z, a))
        
        predicted_set = set(prediction_isotopes)
        
        missing = valid_isotopes_set - predicted_set
        extra = predicted_set - valid_isotopes_set
        
        report = {
            "total_valid": len(valid_isotopes_set),
            "total_predicted": len(predicted_set),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_list": sorted(list(missing)),
            "extra_list": sorted(list(extra)),
            "passed": len(missing) == 0 and len(extra) == 0
        }
        
        return report
