"""
Evaluation metrics for nuclear property prediction.
"""
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for model assessment."""
    
    def __init__(self):
        """Initialize the evaluation metrics calculator."""
        self.results = {}
    
    def calculate_all_metrics(self, y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              target_names: Optional[List[str]] = None) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            target_names: Optional names of targets.
            
        Returns:
            Dictionary of all calculated metrics.
        """
        metrics = {
            'overall': {},
            'per_target': {}
        }
        
        # Overall metrics (averaged across all targets)
        metrics['overall']['mse'] = mean_squared_error(y_true, y_pred)
        metrics['overall']['rmse'] = np.sqrt(metrics['overall']['mse'])
        metrics['overall']['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['overall']['r2'] = r2_score(y_true, y_pred)
        
        # Handle MAPE carefully (avoid division by zero)
        mask = y_true != 0
        if np.any(mask):
            metrics['overall']['mape'] = mean_absolute_percentage_error(
                y_true[mask], y_pred[mask]
            )
        else:
            metrics['overall']['mape'] = np.nan
        
        # Per-target metrics
        n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
        
        for i in range(n_targets):
            target_name = target_names[i] if target_names else f"target_{i}"
            
            y_true_i = y_true[:, i] if len(y_true.shape) > 1 else y_true
            y_pred_i = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
            
            mse = mean_squared_error(y_true_i, y_pred_i)
            
            metrics['per_target'][target_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_true_i, y_pred_i),
                'r2': r2_score(y_true_i, y_pred_i),
                'mean_true': np.mean(y_true_i),
                'mean_pred': np.mean(y_pred_i),
                'std_true': np.std(y_true_i),
                'std_pred': np.std(y_pred_i)
            }
        
        self.results = metrics
        return metrics
    
    def print_report(self, metrics: Optional[Dict] = None) -> str:
        """
        Generate a formatted report of metrics.
        
        Args:
            metrics: Optional metrics dictionary. Uses last calculated if None.
            
        Returns:
            Formatted string report.
        """
        if metrics is None:
            metrics = self.results
        
        if not metrics:
            return "No metrics available. Run calculate_all_metrics first."
        
        report = []
        report.append("=" * 60)
        report.append("EVALUATION METRICS REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\nOVERALL METRICS:")
        report.append("-" * 40)
        for metric_name, value in metrics['overall'].items():
            if isinstance(value, float):
                report.append(f"  {metric_name.upper():8s}: {value:.6f}")
            else:
                report.append(f"  {metric_name.upper():8s}: {value}")
        
        # Per-target metrics
        report.append("\nPER-TARGET METRICS:")
        report.append("-" * 40)
        
        for target_name, target_metrics in metrics['per_target'].items():
            report.append(f"\n  {target_name}:")
            for metric_name, value in target_metrics.items():
                if isinstance(value, float):
                    report.append(f"    {metric_name:12s}: {value:.6f}")
                else:
                    report.append(f"    {metric_name:12s}: {value}")
        
        report.append("\n" + "=" * 60)
        
        report_str = "\n".join(report)
        print(report_str)
        
        return report_str
    
    def compare_predictions(self, y_true: np.ndarray, y_pred1: np.ndarray,
                           y_pred2: np.ndarray,
                           name1: str = "Model 1",
                           name2: str = "Model 2") -> Dict:
        """
        Compare predictions from two models.
        
        Args:
            y_true: True target values.
            y_pred1: Predictions from model 1.
            y_pred2: Predictions from model 2.
            name1: Name for model 1.
            name2: Name for model 2.
            
        Returns:
            Comparison dictionary.
        """
        metrics1 = self.calculate_all_metrics(y_true, y_pred1)
        metrics2 = self.calculate_all_metrics(y_true, y_pred2)
        
        comparison = {
            name1: metrics1['overall'],
            name2: metrics2['overall'],
            'improvement': {}
        }
        
        for metric in metrics1['overall'].keys():
            if metric in metrics2['overall']:
                val1 = metrics1['overall'][metric]
                val2 = metrics2['overall'][metric]
                
                # For R2, higher is better; for others, lower is better
                if metric == 'r2':
                    improvement = val2 - val1
                    direction = "higher"
                else:
                    improvement = val1 - val2
                    direction = "lower"
                
                comparison['improvement'][metric] = {
                    'absolute': improvement,
                    'relative': (improvement / val1 * 100) if val1 != 0 else np.nan,
                    'better_model': name2 if improvement > 0 else name1,
                    'direction': direction
                }
        
        return comparison
