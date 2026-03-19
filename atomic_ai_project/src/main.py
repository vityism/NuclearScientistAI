"""
Main training pipeline for nuclear energy level prediction.

This script orchestrates the complete workflow:
1. Fetch energy level data from IAEA LiveChart API for elements 1-40
2. Preprocess and clean the energy level data
3. Prepare features and energy level sequences
4. Train the LSTM neural network model
5. Evaluate performance
6. Predict energy levels for all isotopes of elements 41-118
7. Output predictions as CSV tables/matrices
"""
import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DEFAULT_TRAINING_START,
    DEFAULT_TRAINING_END,
    DEFAULT_PREDICTION_START,
    DEFAULT_PREDICTION_END,
    MIN_TRAINING_ELEMENTS,
    MAX_TRAINING_ELEMENTS,
    NUCLEAR_FEATURES,
    PROCESSED_DATA_DIR,
    ENERGY_LEVEL_CONFIG
)
from src.data_collection.data_fetcher import DataFetcher
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_loader import DataLoader
from src.model.model_trainer import EnergyLevelTrainer
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.visualizer import PredictionVisualizer


def main(args):
    """Run the complete training and prediction pipeline for energy levels."""
    
    # Determine training range based on CLI argument or defaults
    if args.training_elements is not None:
        training_end = args.training_elements
        # Validate range
        if training_end < MIN_TRAINING_ELEMENTS or training_end > MAX_TRAINING_ELEMENTS:
            print(f"Error: Training elements must be between {MIN_TRAINING_ELEMENTS} and {MAX_TRAINING_ELEMENTS}.")
            print(f"You specified: {training_end}")
            return
        training_start = DEFAULT_TRAINING_START
    else:
        training_start = DEFAULT_TRAINING_START
        training_end = DEFAULT_TRAINING_END
    
    # Set prediction range (elements after training_end up to 118)
    prediction_start = training_end + 1
    prediction_end = DEFAULT_PREDICTION_END
    
    training_elements = list(range(training_start, training_end + 1))
    prediction_elements = list(range(prediction_start, prediction_end + 1))
    
    print("=" * 70)
    print("NUCLEAR ENERGY LEVEL PREDICTION AI")
    print(f"Training on isotopes of elements {training_start}-{training_end} (N={len(training_elements)})")
    print(f"Predicting for elements {prediction_start}-{prediction_end}")
    print("Data source: IAEA LiveChart API (no API key required)")
    print("Output: Energy level tables/matrices for each isotope")
    print("=" * 70)
    
    # Step 1: Fetch data from IAEA LiveChart API
    print(f"\n[STEP 1] Fetching energy level data for elements {training_start}-{training_end}...")
    fetcher = DataFetcher(training_elements=training_elements)
    
    if args.fetch_data:
        raw_filepath = fetcher.fetch_and_save_training_data()
        print(f"Raw data saved to: {raw_filepath}")
    else:
        # Load existing raw data
        loader = DataLoader()
        available_files = loader.get_available_raw_files()
        if available_files:
            raw_filename = available_files[0]  # Use most recent
            print(f"Loading existing raw data: {raw_filename}")
        else:
            print("No raw data found. Please run with --fetch-data flag.")
            return
    
    # Step 2: Load and clean data
    print("\n[STEP 2] Cleaning energy level data...")
    loader = DataLoader()
    raw_data = loader.load_raw_json("training_data_raw.json")
    
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_dataset(raw_data)
    
    # Save cleaned data
    loader.save_processed_csv(cleaned_df, "cleaned_energy_level_data.csv")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Step 3: Prepare features and energy level sequences
    print("\n[STEP 3] Preparing features and energy level sequences...")
    engineer = FeatureEngineer()
    
    # Prepare input features (Z, A, and derived features)
    feature_cols = ['atomic_number', 'mass_number']
    X, feature_names = engineer.prepare_input_features(
        cleaned_df, 
        feature_cols=feature_cols,
        fit=True
    )
    
    # Prepare energy level sequences as targets
    # Shape: [num_isotopes, max_levels, 2] where 2 = [energy, spin_parity_encoded]
    y, isotope_info = engineer.prepare_energy_level_targets(
        cleaned_df,
        max_levels=ENERGY_LEVEL_CONFIG["max_levels_per_isotope"]
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Energy level target shape: {y.shape}")
    print(f"Number of isotopes: {X.shape[0]}")
    print(f"Max energy levels per isotope: {y.shape[1]}")
    print(f"Features: {feature_names}")
    
    # Save processed data
    loader.save_model_data(X, y, feature_names, "energy_level_model_data.pkl")
    
    # Step 4: Train model
    print("\n[STEP 4] Training LSTM neural network model...")
    trainer = EnergyLevelTrainer()
    
    metadata = trainer.train_and_save(
        X=X,
        y=y,
        feature_names=feature_names,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_filename="energy_level_predictor.h5"
    )
    
    # Step 5: Evaluate model
    print("\n[STEP 5] Evaluating model...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42
    )
    
    y_pred = trainer.model.predict(X_val)
    
    # Calculate MSE for energy predictions
    energy_mse = np.mean((y_pred[:, :, 0] - y_val[:, :, 0]) ** 2)
    print(f"\nValidation Energy MSE: {energy_mse:.6f} keV²")
    print(f"Validation Energy RMSE: {np.sqrt(energy_mse):.3f} keV")
    
    # Step 6: Create visualizations
    print("\n[STEP 6] Creating visualizations...")
    viz_dir = os.path.join(PROCESSED_DATA_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    visualizer = PredictionVisualizer(save_dir=viz_dir)
    
    # Plot sample energy level predictions
    visualizer.plot_energy_levels_comparison(
        y_val[:5], y_pred[:5],
        isotope_info=isotope_info[:5],
        save_name="energy_levels_comparison.png"
    )
    
    # Plot training history
    if trainer.model.history:
        visualizer.plot_training_history(
            trainer.model.history,
            save_name="training_history.png"
        )
    
    # Step 7: Predict for remaining elements (training_end+1 to 118)
    print(f"\n[STEP 7] Predicting energy levels for elements {prediction_start}-{prediction_end}...")
    
    # Create features for prediction elements
    # Generate isotopes for each element in prediction range
    prediction_isotopes = []
    for z in prediction_elements:
        # Estimate stable/known isotope range
        min_a = z  # At least Z protons
        max_a = z + int(z * 0.6)  # Approximate neutron-rich limit
        for a in range(min_a, min(max_a, min_a + 15)):  # Limit to ~15 isotopes per element
            prediction_isotopes.append({
                'atomic_number': z,
                'mass_number': a
            })
    
    # Create feature matrix for predictions using the same structure as training
    X_pred_raw, pred_feature_names = engineer.prepare_prediction_features(prediction_isotopes)
    
    # Scale features using fitted scaler
    X_pred_scaled = engineer.scale_prediction_features(X_pred_raw, pred_feature_names)
    
    # Make predictions
    print(f"Predicting for {len(prediction_isotopes)} isotopes...")
    predictions_table = trainer.predict_energy_levels(X_pred_scaled, prediction_isotopes)
    
    # Export predictions to CSV (matrix/table format) - filename reflects dynamic range
    output_filename = f"predicted_energy_levels_{prediction_start}_{prediction_end}.csv"
    output_csv = os.path.join(PROCESSED_DATA_DIR, output_filename)
    trainer.model.export_predictions_to_csv(predictions_table, output_csv)
    
    print(f"\nPredictions exported to: {output_csv}")
    
    # Also save as JSON for programmatic access - filename reflects dynamic range
    output_json_filename = f"predicted_energy_levels_{prediction_start}_{prediction_end}.json"
    output_json = os.path.join(PROCESSED_DATA_DIR, output_json_filename)
    with open(output_json, 'w') as f:
        json.dump(predictions_table, f, indent=2)
    print(f"JSON output saved to: {output_json}")
    
    # Print sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (First 3 isotopes):")
    print("=" * 70)
    for i in range(min(3, len(predictions_table))):
        iso = predictions_table[i]
        print(f"\n{iso['element_symbol']}-{iso['mass_number']} (Z={iso['atomic_number']}):")
        print(f"  {'Level':<6} {'Energy (keV)':<15} {'Spin-Parity':<12}")
        print(f"  {'-'*6} {'-'*15} {'-'*12}")
        for level in iso['energy_levels'][:10]:  # Show first 10 levels
            print(f"  {level['level']:<6} {level['energy_keV']:<15.3f} {level['spin_parity']:<12}")
        if len(iso['energy_levels']) > 10:
            print(f"  ... ({len(iso['energy_levels']) - 10} more levels)")
    
    # Step 8: Save training metadata
    print("\n[STEP 8] Saving training metadata...")
    metadata_file = os.path.join(PROCESSED_DATA_DIR, "training_metadata.json")
    
    # Convert numpy types to Python types for JSON serialization
    metadata_serializable = {}
    for key, value in metadata.items():
        if isinstance(value, (list, str)):
            metadata_serializable[key] = value
        elif hasattr(value, 'item'):
            metadata_serializable[key] = value.item()
        else:
            metadata_serializable[key] = float(value) if isinstance(value, float) else value
    
    metadata_serializable['prediction_summary'] = {
        'total_isotopes_predicted': len(predictions_table),
        'elements_covered': list(set([p['atomic_number'] for p in predictions_table])),
        'output_files': [output_csv, output_json]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("TRAINING AND PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Trained model: {metadata['model_path']}")
    print(f"  - CSV Predictions (table format): {output_csv}")
    print(f"  - JSON Predictions: {output_json}")
    print(f"  - Visualizations: {viz_dir}/")
    print(f"  - Metadata: {metadata_file}")
    print(f"\nTotal isotopes predicted: {len(predictions_table)}")
    print(f"Elements covered: {len(metadata_serializable['prediction_summary']['elements_covered'])} (Z={prediction_start}-{prediction_end})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AI model for nuclear energy level prediction using IAEA LiveChart API"
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch fresh data from IAEA LiveChart API"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--training-elements",
        type=int,
        default=None,
        help=f"Number of first N elements to use for training (must be between {MIN_TRAINING_ELEMENTS} and {MAX_TRAINING_ELEMENTS}). Default is {DEFAULT_TRAINING_END}."
    )
    
    args = parser.parse_args()
    main(args)
