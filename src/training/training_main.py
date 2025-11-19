import argparse
import json
import os
from datetime import datetime
import sys

import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_config import *
from training_plotter import *
from utility import *
from data.training_data.dataset import create_data_loaders

from model.transformer_model import (
    FlareClassifier,
    Trainer,
    WeightedBCELoss,
    get_class_weights,
    train_flare_detection_model
)
from config.model_config import ModelConfig


def train_model(config, device='cuda'):
    """
        Main training function that orchestrates the entire training process.
        
        This function handles data loading, model initialization, training loop execution,
        and result saving according to the methodology described in the thesis.
        
        Arguments:
            config (ModelConfig): Model configuration object from YAML
            device (str): Device to use for training ('cuda' or 'cpu')
            
        Returns:
            tuple: (model, history) The trained model and training history
    """
    # Set random seed
    set_seed(config.seed)
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directories = setup_directories(config.results_dir, timestamp)
    
    # Load data
    print(f"Loading data from {config.processed_dir}...")
    train_loader, val_loader = create_data_loaders(
        processed_dir=config.processed_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Train model
    print(f"Training model on {device}...")
    model, history = train_flare_detection_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=config.input_dim,
        device=device,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        checkpoint_path=str(directories['checkpoints'])
    )
    
    # Create plots
    print("Generating plots...")
    plotter = TrainingPlotter(directories['plots'])
    plotter.plot_training_history(history)
    
    # Load best model for visualization
    best_model_path = directories['checkpoints'] / "best_model.pth"
    best_model = FlareClassifier(
        input_dim=config.input_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    
    # Visualize attention maps
    print("Generating attention visualizations...")
    attention_plotter = TrainingPlotter(directories['attention_maps'])
    attention_plotter.plot_attention_maps(best_model, val_loader)
    
    # Print final metrics
    final_metrics = history['val_metrics'][-1]
    print("\nFinal validation metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    
    # Save configuration and results
    config_dict = {
        'timestamp': timestamp,
        'config': config.to_dict(),
        'final_metrics': final_metrics
    }
    
    with open(directories['run'] / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Save a copy of the configuration YAML
    config.save(directories['run'] / "config.yaml")
    
    print(f"\nTraining complete. Results saved to {directories['run']}")
    
    return model, history



def parse_arguments():
    """
        Parse command line arguments for training configuration.
        
        Returns:
            argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Transformer model for stellar flare detection'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (uses default_config.yaml if not provided)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cuda or cpu)'
    )
    
    return parser.parse_args()


def main():
    """
        Main entry point for the training script.
        
        Returns:
            None
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration from YAML
    print(f"Loading configuration from {args.config if args.config else 'default_config.yaml'}...")
    config = ModelConfig(args.config)
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Model: d_model={config.d_model}, heads={config.num_heads}, layers={config.num_layers}")
    print(f"  Training: batch_size={config.batch_size}, epochs={config.num_epochs}, lr={config.learning_rate}")
    print(f"  Data: {config.processed_dir}")
    print(f"  Device: {args.device}\n")
    
    # Train model
    train_model(config, device=args.device)


if __name__ == '__main__':
    main()