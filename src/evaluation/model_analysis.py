import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings('ignore')


class NFLModelAnalyzer:
    """
    A comprehensive analyzer for NFL prediction models supporting both 
    classification and regression tasks.
    """
    
    def __init__(self, model_type, model_name="NFL Model", target_names=None):
        """
        Initialize the analyzer.
        
        Args:
            model_type (str): Either 'classification' or 'regression'
            model_name (str): Name of your model for reporting
            target_names (list): For classification - list of class names (e.g., ['Loss', 'Win'])
        """
        if model_type not in ['classification', 'regression']:
            raise ValueError("model_type must be either 'classification' or 'regression'")
        
        self.model_type = model_type
        self.model_name = model_name
        self.target_names = target_names
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.results = {}
        
    def set_predictions(self, y_true, y_pred, y_pred_proba=None):
        """
        Set the true values and predictions for analysis.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (for classification only)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        if y_pred_proba is not None:
            self.y_pred_proba = np.array(y_pred_proba)
        
    def analyze_classification_metrics(self):
        """Calculate and store classification metrics."""
        if self.model_type != 'classification':
            print("This method is only available for classification models.")
            return None
            
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        # Basic metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # Store results
        self.results['classification_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        # AUC if probabilities are available
        if self.y_pred_proba is not None:
            try:
                if len(np.unique(self.y_true)) == 2:  # Binary classification
                    auc = roc_auc_score(self.y_true, self.y_pred_proba[:, 1] if self.y_pred_proba.ndim > 1 else self.y_pred_proba)
                else:  # Multi-class
                    auc = roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr', average='weighted')
                self.results['classification_metrics']['auc_roc'] = auc
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
        
        return self.results['classification_metrics']
    
    def analyze_regression_metrics(self):
        """Calculate and store regression metrics."""
        if self.model_type != 'regression':
            print("This method is only available for regression models.")
            return None
            
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_true, self.y_pred)
        
        # Calculate MAPE manually to handle zero values
        mask = self.y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        else:
            mape = np.inf
        
        # Store results
        self.results['regression_metrics'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
        
        return self.results['regression_metrics']
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """Plot confusion matrix for classification models."""
        if self.model_type != 'classification':
            print("Confusion matrix is only available for classification models.")
            return None
            
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        plt.figure(figsize=figsize)
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """Plot ROC curve for binary classification."""
        if self.model_type != 'classification':
            print("ROC curve is only available for classification models.")
            return None
            
        if self.y_pred_proba is None:
            print("ROC curve requires prediction probabilities.")
            return None
        
        if len(np.unique(self.y_true)) != 2:
            print("ROC curve currently supports binary classification only.")
            return None
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1] if self.y_pred_proba.ndim > 1 else self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba[:, 1] if self.y_pred_proba.ndim > 1 else self.y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, figsize=(12, 5), save_path=None):
        """Plot prediction distribution analysis."""
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        if self.model_type == 'classification':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Prediction distribution
            pred_counts = pd.Series(self.y_pred).value_counts().sort_index()
            true_counts = pd.Series(self.y_true).value_counts().sort_index()
            
            x = np.arange(len(pred_counts))
            width = 0.35
            
            ax1.bar(x - width/2, true_counts.values, width, label='True', alpha=0.7)
            ax1.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.7)
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Count')
            ax1.set_title('True vs Predicted Class Distribution')
            ax1.legend()
            if self.target_names:
                ax1.set_xticks(x)
                ax1.set_xticklabels(self.target_names)
            
            # Probability distribution if available
            if self.y_pred_proba is not None:
                if self.y_pred_proba.ndim > 1 and self.y_pred_proba.shape[1] > 1:
                    ax2.hist(self.y_pred_proba[:, 1], bins=20, alpha=0.7, edgecolor='black')
                else:
                    ax2.hist(self.y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Predicted Probability')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Predicted Probability Distribution')
            else:
                ax2.text(0.5, 0.5, 'No probability\ndata available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Probability Distribution')
        
        else:  # Regression
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Actual vs Predicted scatter plot
            ax1.scatter(self.y_true, self.y_pred, alpha=0.6)
            ax1.plot([self.y_true.min(), self.y_true.max()], 
                    [self.y_true.min(), self.y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('True vs Predicted Values')
            
            # Residuals plot
            residuals = self.y_true - self.y_pred
            ax2.scatter(self.y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, top_n=10):
        """Analyze prediction errors."""
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        if self.model_type == 'classification':
            # Find misclassified examples
            errors = self.y_true != self.y_pred
            error_indices = np.where(errors)[0]
            
            if len(error_indices) == 0:
                print("No classification errors found!")
                return {'error_count': 0, 'error_rate': 0.0}
            
            error_analysis = {
                'error_count': len(error_indices),
                'error_rate': len(error_indices) / len(self.y_true),
                'error_indices': error_indices[:top_n],
                'misclassified_pairs': list(zip(self.y_true[error_indices][:top_n], 
                                              self.y_pred[error_indices][:top_n]))
            }
            
        else:  # Regression
            # Calculate absolute errors
            abs_errors = np.abs(self.y_true - self.y_pred)
            error_indices = np.argsort(abs_errors)[-top_n:][::-1]  # Top errors
            
            error_analysis = {
                'mean_absolute_error': np.mean(abs_errors),
                'max_error': np.max(abs_errors),
                'worst_predictions': {
                    'indices': error_indices,
                    'true_values': self.y_true[error_indices],
                    'predicted_values': self.y_pred[error_indices],
                    'absolute_errors': abs_errors[error_indices]
                }
            }
        
        self.results['error_analysis'] = error_analysis
        return error_analysis
    
    def generate_report(self):
        """Generate a comprehensive text report of all analyses."""
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        report = f"\n{'='*60}\n"
        report += f"NFL MODEL ANALYSIS REPORT: {self.model_name}\n"
        report += f"Model Type: {self.model_type.title()}\n"
        report += f"{'='*60}\n\n"
        
        # Dataset info
        report += f"Dataset Information:\n"
        report += f"- Total Samples: {len(self.y_true)}\n"
        
        if self.model_type == 'classification':
            unique_classes = np.unique(self.y_true)
            report += f"- Number of Classes: {len(unique_classes)}\n"
            report += f"- Class Distribution: {dict(zip(*np.unique(self.y_true, return_counts=True)))}\n\n"
            
            # Classification metrics
            if 'classification_metrics' in self.results:
                metrics = self.results['classification_metrics']
                report += f"Classification Metrics:\n"
                report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
                report += f"- Precision: {metrics['precision']:.4f}\n"
                report += f"- Recall: {metrics['recall']:.4f}\n"
                report += f"- F1-Score: {metrics['f1_score']:.4f}\n"
                if 'auc_roc' in metrics:
                    report += f"- AUC-ROC: {metrics['auc_roc']:.4f}\n"
                report += "\n"
        
        else:  # Regression
            report += f"- Target Range: [{self.y_true.min():.2f}, {self.y_true.max():.2f}]\n"
            report += f"- Target Mean: {self.y_true.mean():.2f}\n"
            report += f"- Target Std: {self.y_true.std():.2f}\n\n"
            
            # Regression metrics
            if 'regression_metrics' in self.results:
                metrics = self.results['regression_metrics']
                report += f"Regression Metrics:\n"
                report += f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}\n"
                report += f"- Mean Squared Error (MSE): {metrics['mse']:.4f}\n"
                report += f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n"
                report += f"- R² Score: {metrics['r2_score']:.4f}\n"
                report += f"- Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n\n"
        
        # Error analysis
        if 'error_analysis' in self.results:
            error_analysis = self.results['error_analysis']
            report += f"Error Analysis:\n"
            if self.model_type == 'classification':
                report += f"- Total Errors: {error_analysis['error_count']}\n"
                report += f"- Error Rate: {error_analysis['error_rate']:.4f}\n"
            else:
                report += f"- Mean Absolute Error: {error_analysis['mean_absolute_error']:.4f}\n"
                report += f"- Maximum Error: {error_analysis['max_error']:.4f}\n"
            report += "\n"
        
        report += f"{'='*60}\n"
        report += "Analysis completed successfully!\n"
        report += f"{'='*60}\n"
        
        print(report)
        return report
    
    def full_analysis(self, show_plots=True, save_plots=True, save_dir="./"):
        """
        Run complete analysis including all metrics, plots, and reports.
        
        Args:
            show_plots (bool): Whether to display plots
            save_plots (bool): Whether to save plots to files
            save_dir (str): Directory to save plots
        """
        if self.y_true is None or self.y_pred is None:
            print("Please set predictions first using set_predictions()")
            return None
        
        print(f"Running full analysis for {self.model_name}...")
        
        # Calculate metrics
        if self.model_type == 'classification':
            self.analyze_classification_metrics()
        else:
            self.analyze_regression_metrics()
        
        # Error analysis
        self.analyze_errors()
        
        # Generate plots if requested
        if show_plots or save_plots:
            if self.model_type == 'classification':
                # Confusion Matrix
                save_path = f"{save_dir}confusion_matrix_{self.model_name}.png" if save_plots else None
                self.plot_confusion_matrix(save_path=save_path)
                
                # ROC Curve (if probabilities available)
                if self.y_pred_proba is not None and len(np.unique(self.y_true)) == 2:
                    save_path = f"{save_dir}roc_curve_{self.model_name}.png" if save_plots else None
                    self.plot_roc_curve(save_path=save_path)
            
            # Prediction distribution
            save_path = f"{save_dir}prediction_distribution_{self.model_name}.png" if save_plots else None
            self.plot_prediction_distribution(save_path=save_path)
        
        # Generate report
        # self.generate_report()
        
        return self.results
