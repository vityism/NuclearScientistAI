"""
Main training pipeline for nuclear property prediction.

This script orchestrates the complete workflow:
1. Fetch data from IAEA API for elements 1-40
2. Preprocess and clean the data
3. Engineer features
4. Train the neural network model
5. Evaluate performance
6. Predict properties for elements 41-118
"""
import os
import sys
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TRAINING_ELEMENTS, 
    PREDICTION_ELEMENTS,
    NUCLEAR_FEATURES,
    PROCESSED_DATA_DIR
)
from src.data_collection.data_fetcher import DataFetcher
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_loader import DataLoader
from src.model.model_trainer import ModelTrainer
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.visualizer import PredictionVisualizer


def main(args):
    """Run the complete training and prediction pipeline."""
    
    print("=" * 70)
    print("NUCLEAR PROPERTY PREDICTION AI")
    print("Training on elements 1-40, predicting for elements 41-118")
    print("Data source: IAEA LiveChart API (no API key required)")
    print("=" * 70)
    
    # Step 1: Fetch data from IAEA LiveChart API
    print("\n[STEP 1] Fetching data from IAEA LiveChart API...")
    fetcher = DataFetcher()
    
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
    print("\n[STEP 2] Cleaning data...")
    loader = DataLoader()
    raw_data = loader.load_raw_json("training_data_raw.json")
    
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_dataset(raw_data)
    
    # Save cleaned data
    loader.save_processed_csv(cleaned_df, "cleaned_nuclear_data.csv")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Step 3: Feature engineering
    print("\n[STEP 3] Engineering features...")
    engineer = FeatureEngineer()
    
    # Define target columns
    target_cols = ['binding_energy', 'half_life_seconds', 'neutron_cross_section']
    
    # Prepare features
    X, y, feature_names = engineer.prepare_features(
        cleaned_df, 
        target_cols=target_cols,
        fit=True
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Features: {feature_names}")
    
    # Save processed data
    loader.save_model_data(X, y, feature_names, "model_data.pkl")
    
    # Step 4: Train model
    print("\n[STEP 4] Training neural network model...")
    trainer = ModelTrainer()
    
    metadata = trainer.train_and_save(
        X=X,
        y=y,
        target_names=target_cols,
        feature_names=feature_names,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_filename="nuclear_predictor.h5"
    )
    
    # Step 5: Evaluate model
    print("\n[STEP 5] Evaluating model...")
    evaluator = EvaluationMetrics()
    
    # Get validation predictions
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42
    )
    
    y_pred = trainer.model.predict(X_val)
    metrics = evaluator.calculate_all_metrics(y_val, y_pred, target_cols)
    evaluator.print_report(metrics)
    
    # Step 6: Create visualizations
    print("\n[STEP 6] Creating visualizations...")
    viz_dir = os.path.join(PROCESSED_DATA_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    visualizer = PredictionVisualizer(save_dir=viz_dir)
    
    # Plot predictions vs actual
    visualizer.plot_predictions_vs_actual(
        y_val, y_pred, 
        target_names=target_cols,
        save_name="predictions_vs_actual.png"
    )
    
    # Plot residuals
    visualizer.plot_residuals(
        y_val, y_pred,
        target_names=target_cols,
        save_name="residuals.png"
    )
    
    # Plot training history
    if trainer.model.history:
        visualizer.plot_training_history(
            trainer.model.history,
            save_name="training_history.png"
        )
    
    # Step 7: Predict for remaining elements
    print("\n[STEP 7] Predicting properties for elements 41-118...")
    
    # Create features for prediction elements
    prediction_df = engineer.create_prediction_features(PREDICTION_ELEMENTS)
    
    # Handle missing values in prediction data
    prediction_df = engineer.handle_missing_values(prediction_df)
    
    # Scale features using fitted scaler
    _, X_pred = engineer.scale_features(
        prediction_df, 
        feature_names,
        fit=False
    )
    
    # Make predictions
    predictions = trainer.model.predict(X_pred)
    
    # Save predictions
    prediction_results = {
        'atomic_numbers': PREDICTION_ELEMENTS,
        'predictions': predictions.tolist(),
        'target_names': target_cols
    }
    
    predictions_file = os.path.join(PROCESSED_DATA_DIR, "predictions_elements_41_118.json")
    with open(predictions_file, 'w') as f:
        json.dump(prediction_results, f, indent=2)
    
    print(f"Predictions saved to: {predictions_file}")
    
    # Print sample predictions
    print("\nSample Predictions (first 5 elements):")
    print("-" * 60)
    for i in range(5):
        atomic_num = PREDICTION_ELEMENTS[i]
        print(f"\nElement {atomic_num}:")
        for j, target in enumerate(target_cols):
            print(f"  {target}: {predictions[i][j]:.6f}")
    
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
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("TRAINING AND PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Trained model: {metadata['model_path']}")
    print(f"  - Predictions: {predictions_file}")
    print(f"  - Visualizations: {viz_dir}/")
    print(f"  - Metadata: {metadata_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AI model for nuclear property prediction using IAEA LiveChart API"
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
    
    args = parser.parse_args()
    main(args)
