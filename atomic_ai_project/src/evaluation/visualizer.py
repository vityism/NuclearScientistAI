"""
Visualization tools for nuclear property predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


class PredictionVisualizer:
    """Visualization tools for model predictions and evaluation."""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots. If None, plots are not saved.
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   target_names: List[str] = None,
                                   save_name: str = None) -> plt.Figure:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            target_names: Names of targets.
            save_name: Optional filename to save plot.
            
        Returns:
            Matplotlib figure.
        """
        n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
        
        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            name = target_names[i] if target_names else f"Target {i+1}"
            
            y_true_i = y_true[:, i] if n_targets > 1 else y_true
            y_pred_i = y_pred[:, i] if n_targets > 1 else y_pred
            
            ax.scatter(y_true_i, y_pred_i, alpha=0.6, edgecolors='k')
            
            # Perfect prediction line
            min_val = min(y_true_i.min(), y_pred_i.min())
            max_val = max(y_true_i.max(), y_pred_i.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}\nPredicted vs Actual')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       target_names: List[str] = None,
                       save_name: str = None) -> plt.Figure:
        """
        Plot residual distributions.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            target_names: Names of targets.
            save_name: Optional filename to save plot.
            
        Returns:
            Matplotlib figure.
        """
        n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
        
        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            name = target_names[i] if target_names else f"Target {i+1}"
            
            y_true_i = y_true[:, i] if n_targets > 1 else y_true
            y_pred_i = y_pred[:, i] if n_targets > 1 else y_pred
            
            residuals = y_true_i - y_pred_i
            
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Residual (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nResidual Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        return fig
    
    def plot_training_history(self, history, save_name: str = None) -> plt.Figure:
        """
        Plot training history (loss and metrics over epochs).
        
        Args:
            history: Keras training history object.
            save_name: Optional filename to save plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error Over Training')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                                importance: np.ndarray,
                                top_n: int = 15,
                                save_name: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names.
            importance: Array of importance scores.
            top_n: Number of top features to display.
            save_name: Optional filename to save plot.
            
        Returns:
            Matplotlib figure.
        """
        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        return fig
    
    def plot_element_predictions(self, atomic_numbers: List[int],
                                 predictions: np.ndarray,
                                 target_names: List[str],
                                 save_name: str = None) -> plt.Figure:
        """
        Plot predictions across elements (by atomic number).
        
        Args:
            atomic_numbers: List of atomic numbers.
            predictions: Predicted values.
            target_names: Names of target properties.
            save_name: Optional filename to save plot.
            
        Returns:
            Matplotlib figure.
        """
        n_targets = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4 * n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            name = target_names[i]
            
            pred_i = predictions[:, i] if n_targets > 1 else predictions
            
            ax.plot(atomic_numbers, pred_i, 'bo-', markersize=6, label='Predicted')
            ax.set_xlabel('Atomic Number')
            ax.set_ylabel(name)
            ax.set_title(f'Predicted {name} vs Atomic Number')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        return fig
