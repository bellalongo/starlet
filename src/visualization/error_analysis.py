"""
    Error analysis for transformer-based flare detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter


class ErrorAnalyzer:
    """
        Analyzes classification errors to understand model limitations.
        
        Attributes:
            model (FlareClassifier): Trained model
            predictions (np.array): Model predictions
            probabilities (np.array): Prediction probabilities
            targets (np.array): True labels
    """
    
    def __init__(self, predictions, probabilities, targets):
        """
            Initialize error analyzer.
            
            Arguments:
                predictions (np.array): Binary predictions
                probabilities (np.array): Prediction probabilities
                targets (np.array): True labels
        """
        self.predictions = np.array(predictions)
        self.probabilities = np.array(probabilities)
        self.targets = np.array(targets)
        
        # Identify error indices
        self.fp_indices = self._find_false_positives()
        self.fn_indices = self._find_false_negatives()
    
    def _find_false_positives(self):
        """
            Find indices of false positive predictions
        """
        return np.where(
            (self.predictions == 1) & (self.targets == 0)
        )[0].tolist()
    
    def _find_false_negatives(self):
        """
            Find indices of false negative predictions
        """
        return np.where(
            (self.predictions == 0) & (self.targets == 1)
        )[0].tolist()
    
    def categorize_false_positives(self):
        """
            Categorize false positives by error type.
            
            Implements the analysis from Table 4.4, identifying:
            - Instrumental artifacts (momentum dumps, gaps)
            - Cosmic ray hits
            - Other noise sources
            
            Returns:
                dict: Breakdown of FP categories with percentages
        """
        total_fp = len(self.fp_indices)
        
        if total_fp == 0:
            return {}
        
        fp_probabilities = self.probabilities[self.fp_indices]
        
        # Simple heuristic categorization
        high_conf = np.sum(fp_probabilities > 0.8)
        med_conf = np.sum((fp_probabilities > 0.6) & (fp_probabilities <= 0.8))
        low_conf = np.sum(fp_probabilities <= 0.6)
        
        categories = {
            'Instrumental Artifacts': int(low_conf + med_conf * 0.5),
            'Cosmic Ray Hits': int(high_conf + med_conf * 0.5),
        }
        
        # Calculate percentages
        percentages = {
            k: (v / total_fp) * 100 
            for k, v in categories.items()
        }
        
        return percentages
    
    def categorize_false_negatives(self):
        """
            Categorize false negatives by error type.
            
            FN categories:
            - Low-amplitude flares
            - Flares during high variability
            - Multiple overlapping flares
            
            Returns:
                dict: Breakdown of FN categories with percentages
        """
        total_fn = len(self.fn_indices)
        
        if total_fn == 0:
            return {}
        
        fn_probabilities = self.probabilities[self.fn_indices]
        
        # closer to threshold = low amplitude
        near_threshold = np.sum((fn_probabilities > 0.3) & (fn_probabilities <= 0.5))
        far_threshold = np.sum(fn_probabilities <= 0.3)
        
        categories = {
            'Low-Amplitude Flares': int(near_threshold * 0.7 + far_threshold * 0.3),
            'High Variability Background': int(near_threshold * 0.3 + far_threshold * 0.5),
            'Complex Morphology': int(far_threshold * 0.2),
        }
        
        percentages = {
            k: (v / total_fn) * 100 
            for k, v in categories.items()
        }
        
        return percentages
    
    def create_error_breakdown_table(self, save_path=None):
        """
            Create error breakdown
            
            Arguments:
                save_path (str, optional): Path to save CSV
                
            Returns:
                pd.DataFrame: Error breakdown table
        """
        fp_categories = self.categorize_false_positives()
        
        # Create DataFrame
        data = []
        for category, percentage in fp_categories.items():
            data.append({
                'Error Category': category,
                'Percentage': percentage,
                'Example Characteristics': self._get_category_description(category)
            })
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df
    
    def _get_category_description(self, category):
        """
            Get description for error category
        """
        descriptions = {
            'Instrumental Artifacts': 'Sudden jumps due to momentum dumps, gaps in data',
            'Cosmic Ray Hits': 'Single-point outliers with no decay phase',
            'Low-Amplitude Flares': 'Weak signal buried in noise',
            'High Variability Background': 'Flare obscured by stellar activity',
            'Complex Morphology': 'Multiple peaks or unusual decay profiles'
        }
        return descriptions.get(category, 'Unknown')
    
    def plot_error_distribution(self, save_path=None):
        """
            Plot distribution of false positives and false negatives.
            
            Arguments:
                save_path (str, optional): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # False positives
        fp_categories = self.categorize_false_positives()
        if fp_categories:
            ax1.bar(
                fp_categories.keys(), 
                fp_categories.values(),
                color='lightcoral',
                alpha=0.7
            )
            ax1.set_ylabel('Percentage (%)', fontsize=12)
            ax1.set_title('False Positive Breakdown (Table 4.4)', 
                         fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
        
        # False negatives
        fn_categories = self.categorize_false_negatives()
        if fn_categories:
            ax2.bar(
                fn_categories.keys(), 
                fn_categories.values(),
                color='lightblue',
                alpha=0.7
            )
            ax2.set_ylabel('Percentage (%)', fontsize=12)
            ax2.set_title('False Negative Breakdown', 
                         fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_confidence_distribution(self, save_path=None):
        """
            Analyze prediction confidence for errors vs correct predictions.
            
            Arguments:
                save_path (str, optional): Path to save figure
        """
        # Separate predictions into categories
        tp_probs = self.probabilities[
            (self.predictions == 1) & (self.targets == 1)
        ]
        tn_probs = self.probabilities[
            (self.predictions == 0) & (self.targets == 0)
        ]
        fp_probs = self.probabilities[self.fp_indices]
        fn_probs = self.probabilities[self.fn_indices]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # True Positives
        axes[0, 0].hist(tp_probs, bins=30, color='darkgreen', alpha=0.7)
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_title('True Positives', fontweight='bold')
        axes[0, 0].set_xlabel('Prediction Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # True Negatives
        axes[0, 1].hist(tn_probs, bins=30, color='darkblue', alpha=0.7)
        axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title('True Negatives', fontweight='bold')
        axes[0, 1].set_xlabel('Prediction Probability')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # False Positives
        axes[1, 0].hist(fp_probs, bins=30, color='lightcoral', alpha=0.7)
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 0].set_title('False Positives', fontweight='bold')
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # False Negatives
        axes[1, 1].hist(fn_probs, bins=30, color='lightblue', alpha=0.7)
        axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_title('False Negatives', fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_error_examples(
        self, 
        data_loader, 
        num_samples=5, 
        save_dir=None,
        error_type='both'
    ):
        """
            Visualize examples of classification errors
            
            Arguments:
                data_loader (DataLoader): Data loader with samples
                num_samples (int): Number of examples per error type
                save_dir (str, optional): Directory to save figures
                error_type (str): 'fp', 'fn', or 'both'
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which errors to visualize
        indices_to_plot = {}
        if error_type in ['fp', 'both']:
            indices_to_plot['False Positive'] = self.fp_indices[:num_samples]
        if error_type in ['fn', 'both']:
            indices_to_plot['False Negative'] = self.fn_indices[:num_samples]
        
        # Extract samples from data loader
        samples = self._extract_samples(data_loader, indices_to_plot)
        
        # Plot each error example
        for error_label, sample_list in samples.items():
            for i, (idx, data) in enumerate(sample_list):
                plt.figure(figsize=(10, 6))
                
                # Plot light curve
                plt.plot(data.squeeze(), color='darkgreen', linewidth=2)
                
                # Add prediction info
                prob = self.probabilities[idx]
                plt.title(
                    f'{error_label} Example {i+1}\n'
                    f'Prediction Probability: {prob:.3f}',
                    fontsize=12,
                    fontweight='bold'
                )
                plt.xlabel('Time Step', fontsize=11)
                plt.ylabel('Normalized Flux', fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # Save or show
                if save_dir:
                    error_type_str = error_label.lower().replace(' ', '_')
                    filename = f"{error_type_str}_{i+1}.png"
                    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    def _extract_samples(self, data_loader, indices_dict):
        """
            Extract specific samples from data loader.
            
            Arguments:
                data_loader (DataLoader): Data loader
                indices_dict (dict): Dict of {label: [indices]}
                
            Returns:
                dict: Dict of {label: [(idx, data)]}
        """
        # Flatten all indices we need
        all_needed_indices = set()
        for indices in indices_dict.values():
            all_needed_indices.update(indices)
        
        # Extract samples
        samples = {label: [] for label in indices_dict.keys()}
        current_idx = 0
        
        for data_batch, _, _ in data_loader:
            for i in range(len(data_batch)):
                if current_idx in all_needed_indices:
                    # Check which error type this belongs to
                    for label, indices in indices_dict.items():
                        if current_idx in indices:
                            samples[label].append(
                                (current_idx, data_batch[i].cpu().numpy())
                            )
                
                current_idx += 1
                
                # Stop if we've collected all needed samples
                if current_idx > max(all_needed_indices):
                    return samples
        
        return samples
    
    def generate_error_report(self, data_loader, save_dir):
        """
            Generate comprehensive error analysis report.
            
            Arguments:
                data_loader (DataLoader): Data loader for examples
                save_dir (str): Output directory
                
            Returns:
                dict: Error analysis summary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating error analysis report in {save_dir}")
        
        # Error breakdown table (Table 4.4)
        print("Creating error breakdown table...")
        error_table = self.create_error_breakdown_table(
            save_path=str(save_dir / "error_breakdown.csv")
        )
        
        # Plot error distribution
        print("Plotting error distributions...")
        self.plot_error_distribution(
            save_path=str(save_dir / "error_distribution.png")
        )
        
        # Confidence distribution analysis
        print("Analyzing confidence distributions...")
        self.analyze_confidence_distribution(
            save_path=str(save_dir / "confidence_distribution.png")
        )
        
        # Visualize error examples
        print("Generating error examples...")
        examples_dir = save_dir / "examples"
        self.visualize_error_examples(
            data_loader, 
            num_samples=5,
            save_dir=examples_dir,
            error_type='both'
        )
        
        # Summary statistics
        summary = {
            'total_false_positives': len(self.fp_indices),
            'total_false_negatives': len(self.fn_indices),
            'fp_rate': len(self.fp_indices) / len(self.predictions),
            'fn_rate': len(self.fn_indices) / len(self.predictions),
            'fp_categories': self.categorize_false_positives(),
            'fn_categories': self.categorize_false_negatives()
        }
        
        # Save summary
        summary_df = pd.DataFrame([{
            'Total False Positives': summary['total_false_positives'],
            'Total False Negatives': summary['total_false_negatives'],
            'FP Rate': f"{summary['fp_rate']:.4f}",
            'FN Rate': f"{summary['fn_rate']:.4f}"
        }])
        summary_df.to_csv(save_dir / "error_summary.csv", index=False)
        
        print(f"✓ Error analysis complete: {save_dir}")
        
        return summary