# Nuclear Energy Level Prediction AI

An AI system trained exclusively on atomic/nuclear characteristics (specifically **energy levels**) of the first 40 elements of the periodic table to predict energy level properties of remaining elements (41-118). Data is sourced exclusively from the IAEA LiveChart API (no API key required).

## Project Structure

```
atomic_ai_project/
├── config/                 # Configuration settings
│   └── settings.py
├── data/                   # Data storage
│   ├── raw/               # Raw data from IAEA LiveChart API
│   └── processed/         # Processed and cleaned data
├── src/                    # Source code
│   ├── data_collection/   # IAEA LiveChart API client and data fetching
│   ├── preprocessing/     # Data cleaning and feature engineering
│   ├── model/             # LSTM neural network model and training
│   ├── evaluation/        # Metrics and visualization
│   └── main.py            # Main training pipeline
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

## Features

- **Data Collection**: Fetches atomic/nuclear data from IAEA LiveChart API including:
  - Excitation energy levels (ground state and excited states)
  - Spin-parity assignments (Jπ) for each energy level
  - Half-life information
  - Decay modes
  - Isotopic data for all isotopes of each element

- **Preprocessing**: 
  - Data cleaning and validation
  - Feature engineering (atomic number Z, mass number A, periodic table position)
  - Spin-parity encoding for neural network input
  - Missing value handling
  - Feature scaling and normalization

- **Model**: LSTM (Long Short-Term Memory) neural network with:
  - Sequential architecture optimized for energy level prediction
  - Multiple hidden layers with dropout regularization
  - Batch normalization for stable training
  - Early stopping to prevent overfitting
  - Learning rate scheduling
  - Output format: Matrix/table structure with energy levels per isotope

- **Evaluation**: Comprehensive metrics including:
  - MSE, RMSE, MAE for energy predictions
  - R² score for model fit quality
  - Per-isotope and per-element analysis
  - Visualization tools for prediction accuracy

## Installation

### Linux (Ubuntu/Debian/CentOS/Fedora)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd atomic_ai_project
```

2. **Install Python dependencies:**
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
```

### Windows 11

1. **Clone the repository:**
```cmd
git clone <repository-url>
cd atomic_ai_project
```

2. **Install Python dependencies:**
```cmd
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

3. **Verify installation:**
```cmd
python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
```

**Note**: No API key is required! The project uses the public IAEA LiveChart API.

## Usage

### Training the Model

#### Linux

To fetch fresh data from IAEA LiveChart API and train the model:

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Fetch data and train with default settings
python src/main.py --fetch-data --epochs 100 --batch-size 32

# Or specify custom parameters
python src/main.py --fetch-data --epochs 200 --batch-size 64 --val-split 0.15
```

#### Windows 11

```cmd
# Activate virtual environment if not already active
venv\Scripts\activate

# Fetch data and train with default settings
python src\main.py --fetch-data --epochs 100 --batch-size 32

# Or specify custom parameters
python src\main.py --fetch-data --epochs 200 --batch-size 64 --val-split 0.15
```

### Using Existing Data

If you have existing raw data (skip data fetching):

**Linux:**
```bash
source venv/bin/activate
python src/main.py --epochs 100 --batch-size 32
```

**Windows 11:**
```cmd
venv\Scripts\activate
python src\main.py --epochs 100 --batch-size 32
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fetch-data` | Fetch fresh data from IAEA LiveChart API | Off |
| `--epochs` | Number of training epochs/generations | 100 |
| `--batch-size` | Training batch size | 32 |
| `--val-split` | Validation split ratio (0.0-1.0) | 0.2 |
| `--learning-rate` | Initial learning rate | 0.001 |
| `--model-path` | Path to save/load model | `models/nuclear_predictor.h5` |

## Customizing Training Time and Generations

The training process can be customized to balance between training time and model accuracy:

### Adjusting Number of Epochs (Generations)

The `--epochs` parameter controls how many times the model sees the entire training dataset:

```bash
# Quick test run (fast, lower accuracy)
python src/main.py --epochs 10

# Standard training (balanced)
python src/main.py --epochs 100

# Extended training (slower, higher accuracy)
python src/main.py --epochs 500

# Maximum precision training (slowest, best accuracy)
python src/main.py --epochs 1000
```

**Estimated Training Times** (varies by hardware):
- **CPU only**: ~2-5 minutes per 100 epochs
- **GPU available**: ~30-60 seconds per 100 epochs

### Adjusting Batch Size

The `--batch-size` parameter affects memory usage and training speed:

```bash
# Smaller batches (more memory efficient, slower convergence)
python src/main.py --batch-size 16 --epochs 200

# Larger batches (faster training, requires more RAM/VRAM)
python src/main.py --batch-size 128 --epochs 50
```

### Adjusting Validation Split

The `--val-split` parameter controls how much data is reserved for validation:

```bash
# More training data (less validation)
python src/main.py --val-split 0.1 --epochs 150

# More validation data (better evaluation)
python src/main.py --val-split 0.3 --epochs 150
```

### Advanced Configuration

Edit `config/settings.py` for advanced customization:

```python
# In config/settings.py
MODEL_CONFIG = {
    'hidden_layers': [128, 64, 32],  # Adjust network size
    'dropout_rate': 0.3,              # Regularization strength
    'early_stopping_patience': 20,    # Stop if no improvement for N epochs
    'max_levels_per_isotope': 50,     # Maximum energy levels to predict
}
```

## Evaluation

### How Evaluation Works

The evaluation system assesses model performance using a hold-out validation approach:

#### Data Used for Evaluation

1. **Training Data**: Elements 1-40 (Hydrogen to Zirconium) - used to train the model
2. **Validation Data**: Randomly selected 20% (configurable via `--val-split`) of the training data is held out during training
3. **Test Data**: Elements 41-118 - predictions are made but cannot be evaluated against ground truth (since these are the unknown targets)

**Important**: The model is ONLY trained on elements 1-40. Evaluation metrics are calculated on the held-out validation portion of elements 1-40, NOT on elements 41-118 (which have no ground truth in this experimental setup).

#### Scoring Metrics

The evaluation calculates the following metrics for each predicted property:

1. **Mean Squared Error (MSE)**
   - Formula: `MSE = (1/n) * Σ(actual - predicted)²`
   - Interpretation: Lower is better; penalizes large errors more heavily
   - Units: keV² for energy levels

2. **Root Mean Squared Error (RMSE)**
   - Formula: `RMSE = √MSE`
   - Interpretation: Lower is better; in same units as target (keV)
   - Example: RMSE of 50 keV means average prediction error is ~50 keV

3. **Mean Absolute Error (MAE)**
   - Formula: `MAE = (1/n) * Σ|actual - predicted|`
   - Interpretation: Lower is better; average absolute deviation
   - More robust to outliers than RMSE

4. **R² Score (Coefficient of Determination)**
   - Formula: `R² = 1 - (SS_res / SS_tot)`
   - Range: -∞ to 1.0
   - Interpretation:
     - R² = 1.0: Perfect prediction
     - R² = 0.0: Model performs no better than predicting the mean
     - R² < 0: Model performs worse than predicting the mean
   - Target: R² > 0.7 indicates good predictive power

#### Per-Target Analysis

Each nuclear property is evaluated separately:
- **Energy Levels**: Primary focus (ground state + excited states up to 50 levels)
- **Spin-Parity**: Encoded categorical predictions
- **Half-life**: Log-transformed continuous values

#### Evaluation Output

After training completes, the system generates:
- **Numerical metrics** printed to console
- **Visualization plots** in `data/processed/visualizations/`:
  - Predicted vs Actual scatter plots
  - Residual distribution histograms
  - Training/validation loss curves
  - Per-element error breakdown
- **Detailed JSON report** with all metrics

### Running Evaluation Independently

```bash
# Evaluate a pre-trained model
python src/main.py --evaluate --model-path models/nuclear_predictor.h5
```

## Output Format

After training and prediction, outputs are generated in multiple formats:

### 1. CSV Table/Matrix (`data/processed/predicted_energy_levels_41_118.csv`)

Structured as a table with columns:
```
atomic_number,element,mass_number,level,energy_keV,spin_parity
41,Nb,93,0,0.0,9/2+
41,Nb,93,1,30.77,1/2-
41,Nb,93,2,103.45,7/2+
...
```

Each row represents one energy level of one isotope. All isotopes are predicted individually.

### 2. JSON Structured Data (`data/processed/predicted_energy_levels_41_118.json`)

Hierarchical format organized by element → isotope → energy levels:
```json
{
  "element_41": {
    "Nb-93": {
      "levels": [
        {"level": 0, "energy_keV": 0.0, "spin_parity": "9/2+"},
        {"level": 1, "energy_keV": 30.77, "spin_parity": "1/2-"},
        ...
      ]
    },
    "Nb-94": {...}
  }
}
```

### 3. Visualization Files

- `predictions_vs_actual.png`: Scatter plot of predicted vs actual values
- `residuals_distribution.png`: Histogram of prediction errors
- `training_history.png`: Loss and accuracy curves over epochs
- `per_element_errors.png`: Error breakdown by element

## Nuclear Properties Predicted

The model predicts the following properties for **each isotope** of elements 41-118:

1. **Excitation Energy Levels** (keV)
   - Ground state (level 0) always at 0 keV
   - Up to 50 excited states per isotope
   - Includes energy value and spin-parity assignment

2. **Spin-Parity (Jπ)**
   - Quantum mechanical angular momentum and parity
   - Encoded format for prediction, decoded for output

3. **Isotope-Specific Predictions**
   - Each isotope (e.g., Nb-93, Nb-94, Nb-95...) predicted independently
   - ~15-20 isotopes per element on average
   - Total: ~1,200-1,500 individual isotope predictions

## Dependencies

- TensorFlow/Keras >= 2.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Requests >= 2.25.0

## Data Source

This project **exclusively** uses the **IAEA LiveChart API** (https://www-nds.iaea.org/relnsd/vchar/), which is a public API that does not require authentication. 

- **Training Data**: All nuclear properties (especially energy levels) from elements 1-40 (Hydrogen to Zirconium), including all known isotopes
- **Prediction Targets**: Elements 41-118 (Niobium to Oganesson), all isotopes predicted individually
- **Data Fields**: Excitation energies, spin-parity assignments, half-lives, decay modes

The IAEA (International Atomic Energy Agency) LiveChart database is the authoritative source for nuclear structure data worldwide.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

**TensorFlow Import Error:**
```bash
pip install --upgrade tensorflow
```

**Memory Error during Training:**
```bash
# Reduce batch size
python src/main.py --batch-size 16 --epochs 200
```

**API Connection Timeout:**
- Check internet connection
- The IAEA LiveChart API may experience temporary downtime
- Try again later or use existing cached data (omit `--fetch-data`)

**Slow Training on CPU:**
- Consider reducing epochs or using Google Colab with GPU
- Use smaller batch sizes to reduce memory pressure
