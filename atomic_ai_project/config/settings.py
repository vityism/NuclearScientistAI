"""
Configuration settings for the Atomic AI Project.
"""
import os

# IAEA LiveChart API Configuration
# The IAEA LiveChart API does not require an API key
IAEA_LIVECHART_BASE_URL = "https://www-nds.iaea.org/relnsd/vchar?tab=nuclide&Z={}&A={}"
IAEA_API_KEY = None  # Not required for LiveChart API

# Data Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model Configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TRAINING_ELEMENTS = list(range(1, 41))  # First 40 elements (Hydrogen to Zirconium)
PREDICTION_ELEMENTS = list(range(41, 119))  # Remaining elements

# Nuclear characteristics to predict
NUCLEAR_FEATURES = [
    "atomic_number",
    "mass_number",
    "binding_energy",
    "half_life",
    "decay_mode",
    "spin_parity",
    "neutron_cross_section",
    "abundance"
]

# Hyperparameters
HYPERPARAMETERS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.2
}
