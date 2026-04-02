#!/usr/bin/env python3
"""
Verification script to extract and display all energy levels from IAEA API.
Outputs both JSON and CSV formats for manual verification.
"""
import sys
import os
import json
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.iaea_client import IAEAClient


def verify_element(element_z, output_dir="./verification_output"):
    """Fetch and verify energy levels for a specific element."""
    client = IAEAClient()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION FOR ELEMENT {element_z}")
    print(f"{'='*80}\n")
    
    # Get valid isotopes
    valid_masses = client.get_valid_isotopes(element_z)
    symbol = client._get_symbol(element_z)
    
    print(f"Element: {symbol} (Z={element_z})")
    print(f"Valid isotopes (mass numbers): {valid_masses}")
    print(f"Total isotopes: {len(valid_masses)}\n")
    
    all_data = {
        "element": symbol,
        "atomic_number": element_z,
        "isotopes": []
    }
    
    # Fetch energy levels for each isotope
    for mass in valid_masses:
        levels = client.get_energy_levels(element_z, mass)
        
        isotope_data = {
            "mass_number": mass,
            "level_count": len(levels),
            "levels": levels
        }
        all_data["isotopes"].append(isotope_data)
        
        # Print summary
        if len(levels) <= 5:
            print(f"  {symbol}-{mass}: {len(levels)} levels")
        else:
            energies_sample = [l['energy_keV'] for l in levels[:5]]
            print(f"  {symbol}-{mass}: {len(levels)} levels - First 5: {energies_sample}")
    
    # Save to JSON
    json_file = os.path.join(output_dir, f"element_{element_z}_verification.json")
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✓ JSON saved to: {json_file}")
    
    # Save to CSV (one row per energy level)
    csv_file = os.path.join(output_dir, f"element_{element_z}_verification.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['element', 'atomic_number', 'mass_number', 'level_index', 
                        'energy_keV', 'spin_parity', 'half_life_sec'])
        
        # Data rows
        for iso in all_data["isotopes"]:
            for idx, level in enumerate(iso["levels"]):
                writer.writerow([
                    symbol,
                    element_z,
                    iso["mass_number"],
                    idx,
                    level.get('energy_keV', ''),
                    level.get('spin_parity', ''),
                    level.get('half_life_sec', '')
                ])
    
    print(f"✓ CSV saved to: {csv_file}")
    
    # Summary statistics
    total_levels = sum(iso["level_count"] for iso in all_data["isotopes"])
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total isotopes: {len(valid_masses)}")
    print(f"  Total energy levels extracted: {total_levels}")
    print(f"  Average levels per isotope: {total_levels/len(valid_masses):.1f}")
    print(f"{'='*80}\n")
    
    return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify energy level extraction from IAEA API")
    parser.add_argument("-z", "--element", type=int, default=24,
                       help="Atomic number of element to verify (default: 24)")
    parser.add_argument("-o", "--output", type=str, default="./verification_output",
                       help="Output directory for verification files")
    
    args = parser.parse_args()
    
    data = verify_element(args.element, args.output)
