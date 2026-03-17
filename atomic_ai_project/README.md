# Nuclear Property Prediction AI

An AI system trained on atomic/nuclear characteristics of the first 40 elements of the periodic table to predict properties of remaining elements (41-118). Data is sourced exclusively from the IAEA atomic database API.

## Project Structure

```
atomic_ai_project/
├── config/                 # Configuration settings
│   └── settings.py
├── data/                   # Data storage
│   ├── raw/               # Raw data from IAEA API
│   └── processed/         # Processed and cleaned data
├── src/                    # Source code
│   ├── data_collection/   # IAEA API client and data fetching
│   ├── preprocessing/     # Data cleaning and feature engineering
│   ├── model/             # Neural network model and training
│   ├── evaluation/        # Metrics and visualization
│   └── main.py            # Main training pipeline
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

## Features

- **Data Collection**: Fetches atomic/nuclear data from IAEA API including:
  - Binding energy
  - Half-life
  - Decay modes
  - Spin-parity
  - Neutron cross-section
  - Isotopic abundance

- **Preprocessing**: 
  - Data cleaning and validation
  - Feature engineering (periodic table position, derived features)
  - Missing value handling
  - Feature scaling

- **Model**: Deep neural network with:
  - Multiple hidden layers
  - Dropout regularization
  - Batch normalization
  - Early stopping
  - Learning rate scheduling

- **Evaluation**: Comprehensive metrics including:
  - MSE, RMSE, MAE
  - R² score
  - Per-target analysis
  - Visualization tools

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd atomic_ai_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variable for IAEA API key (optional):
```bash
export IAEA_API_KEY="your-api-key-here"
```

## Usage

### Training the Model

To fetch fresh data from IAEA API and train the model:

```bash
python src/main.py --fetch-data --epochs 100 --batch-size 32
```

### Using Existing Data

If you have existing raw data:

```bash
python src/main.py --epochs 100
```

### Command Line Options

- `--api-key`: IAEA API key (or set IAEA_API_KEY environment variable)
- `--fetch-data`: Fetch fresh data from IAEA API
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--val-split`: Validation split ratio (default: 0.2)

## Running Tests

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest discover tests/
```

## Output

After training, the following outputs are generated:

- **Trained model**: `models/nuclear_predictor.h5`
- **Predictions**: `data/processed/predictions_elements_41_118.json`
- **Visualizations**: `data/processed/visualizations/`
- **Metadata**: `data/processed/training_metadata.json`

## Nuclear Properties Predicted

The model predicts the following properties for elements 41-118:

1. **Binding Energy** (MeV)
2. **Half-life** (seconds)
3. **Neutron Cross-section** (barns)

## Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Requests

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
