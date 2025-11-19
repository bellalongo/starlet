"""
    Evaluation metrics for transformer-based flare detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix
)
import pandas as pd
from pathlib import Path


class MetricsEvaluator:
    """
        Computes and visualizes evaluation metrics for flare detection.
        
        Implements metrics:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * Precision * Recall / (Precision + Recall)
        
        Attributes:
            model (FlareClassifier): Trained model
            data_loader (DataLoader): Data for evaluation
            device (torch.device): Computation device
            all_predictions (list): Binary predictions
            all_probabilities (list): Prediction probabilities
            all_targets (list): True labels
    """
    
    def __init__(self, model, data_loader, device):
        """
            Initialize evaluator and collect predictions.
            
            Arguments:
                model (FlareClassifier): Trained model
                data_loader (DataLoader): Evaluation data
                device (torch.device): Device for computation
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # Storage for predictions and targets
        self.all_predictions = []
        self.all_probabilities = []
        self.all_targets = []
        
        # Collect all predictions
        self._collect_predictions()
    
    def _collect_predictions(self):
        """
            Collect model predictions on the dataset
        """
        self.model.eval()
        
        with torch.no_grad():
            for data, targets, _ in self.data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get model outputs
                outputs = self.model(data)
                probabilities = outputs.squeeze()
                predictions = (probabilities > 0.5).float()
                
                # Store results
                self.all_probabilities.extend(probabilities.cpu().numpy())
                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_targets.extend(targets.cpu().numpy())
    
    def compute_metrics(self):
        """
            Compute classification metrics
            
            Returns:
                dict: Dictionary containing all metrics
        """
        # Convert to numpy arrays
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate metrics using thesis formulations
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall) 
            if (precision + recall) > 0 
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def plot_roc_curve(self, save_path=None, compare_methods=None):
        """
            Plot ROC curve
            
            Arguments:
                save_path (str, optional): Path to save figure
                compare_methods (dict, optional): Dict of {name: (y_true, y_score)}
                    for comparison with other methods
            
            Returns:
                float: Area under ROC curve 
        """
        y_true = np.array(self.all_targets)
        y_score = np.array(self.all_probabilities)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, 
            color='darkgreen', 
            lw=2, 
            label=f'Transformer (AUC = {roc_auc:.2f})'
        )
        
        # Add comparison methods if provided
        if compare_methods:
            colors = ['lightgreen', 'salmon', 'lightpink', 'pink']
            for i, (name, (y_t, y_s)) in enumerate(compare_methods.items()):
                fpr_comp, tpr_comp, _ = roc_curve(y_t, y_s)
                auc_comp = auc(fpr_comp, tpr_comp)
                plt.plot(
                    fpr_comp, tpr_comp,
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f'{name} (AUC = {auc_comp:.2f})'
                )
        
        # Reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Flare Detection Methods')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, save_path=None):
        """
            Plot precision-recall curve for imbalanced dataset evaluation.
            
            Arguments:
                save_path (str, optional): Path to save figure
            
            Returns:
                float: Area under PR curve
        """
        y_true = np.array(self.all_targets)
        y_score = np.array(self.all_probabilities)
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, 
            color='blue', 
            lw=2, 
            label=f'PR curve (AUC = {pr_auc:.2f})'
        )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return pr_auc
    
    def plot_confusion_matrix(self, save_path=None):
        """
            Plot confusion matrix visualization.
            
            Arguments:
                save_path (str, optional): Path to save figure
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Flare', 'Flare'],
            yticklabels=['No Flare', 'Flare'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_comparison_table(self, baseline_metrics=None, save_path=None):
        """
            Create performance comparison table
            
            Arguments:
                baseline_metrics (dict): Dict of {method_name: metrics_dict}
                save_path (str, optional): Path to save CSV
                
            Returns:
                pd.DataFrame: Comparison table
        """
        # Get transformer metrics
        metrics = self.compute_metrics()
        y_score = np.array(self.all_probabilities)
        y_true = np.array(self.all_targets)
        
        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create table data
        table_data = {
            'Method': ['Transformer'],
            'Accuracy': [metrics['accuracy']],
            'Precision': [metrics['precision']],
            'Recall': [metrics['recall']],
            'F1 Score': [metrics['f1']],
            'AUC-ROC': [roc_auc]
        }
        
        # Add baseline methods if provided
        if baseline_metrics:
            for method, m in baseline_metrics.items():
                table_data['Method'].append(method)
                table_data['Accuracy'].append(m.get('accuracy', 0))
                table_data['Precision'].append(m.get('precision', 0))
                table_data['Recall'].append(m.get('recall', 0))
                table_data['F1 Score'].append(m.get('f1', 0))
                table_data['AUC-ROC'].append(m.get('auc_roc', 0))
        
        df = pd.DataFrame(table_data)
        
        if save_path:
            df.to_csv(save_path, index=False, float_format='%.4f')
        
        return df
    
    def analyze_by_energy(
        self, 
        energy_data=None, 
        energy_bins=5, 
        save_path=None
    ):
        """
            Analyze detection rate by flare energy.
            
            Arguments:
                energy_data (np.array, optional): Flare energies
                energy_bins (int): Number of energy bins
                save_path (str, optional): Path to save figure
                
            Returns:
                pd.DataFrame: Detection rates by energy bin
        """
        print("Note: Energy-based analysis requires flare energy data.")
        print("Using placeholder implementation with synthetic data.")
        
        # Get flare indices
        flare_indices = [
            i for i, t in enumerate(self.all_targets) if t == 1
        ]
        flare_predictions = [self.all_predictions[i] for i in flare_indices]
        
        # Generate synthetic energies (log-normal distribution)
        np.random.seed(42)
        if energy_data is None:
            synthetic_energies = np.exp(np.random.normal(30, 2, len(flare_indices)))
        else:
            synthetic_energies = energy_data
        
        # Create DataFrame
        df = pd.DataFrame({
            'energy': synthetic_energies,
            'detected': flare_predictions
        })
        
        # Create logarithmic bins
        df['log_energy'] = np.log10(df['energy'])
        bins = np.linspace(
            df['log_energy'].min(), 
            df['log_energy'].max(), 
            energy_bins + 1
        )
        df['energy_bin'] = pd.cut(df['log_energy'], bins=bins)
        
        # Calculate detection rate
        detection_rate = df.groupby('energy_bin')['detected'].mean()
        counts = df.groupby('energy_bin').size()
        
        # Plot
        plt.figure(figsize=(10, 6))
        ax = detection_rate.plot(
            kind='bar',
            yerr=np.sqrt(detection_rate * (1 - detection_rate) / counts),
            color='skyblue',
            alpha=0.7
        )
        
        plt.xlabel('Flare Energy Bin (log10 erg)', fontsize=12)
        plt.ylabel('Detection Rate', fontsize=12)
        plt.title('Detection Rate by Flare Energy', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Return results
        bin_labels = [f"{10**bins[i]:.1e}-{10**bins[i+1]:.1e}" 
                      for i in range(len(bins)-1)]
        result_df = pd.DataFrame({
            'energy_bin': bin_labels,
            'detection_rate': detection_rate.values,
            'count': counts.values
        })
        
        return result_df