# Stellar Flare Detection with Transformers

### Read my thesis [here](thesis.pdf) !




[![PyTorch](https://img.shields.io/badge/PyTorch-pink?style=for-the-badge)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-pink?style=for-the-badge)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-pink?style=for-the-badge)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-pink?style=for-the-badge)](https://seaborn.pydata.org/)

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Output Files](#output-files)

## About

A transformer-based deep learning model for detecting stellar flares in TESS light curve data. Stellar flares are sudden bursts of energy from magnetic reconnection on stellar surfaces—this model uses self-attention mechanisms to catch them automatically.

### How It Works

The model performs flare detection through the following steps:

1. **Data Processing**: Loads TESS light curves and extracts 100-timestep windows
2. **Feature Encoding**: Transforms flux, gradient, and quality features through learned embeddings
3. **Attention Analysis**: Multi-head self-attention focuses on characteristic flare signatures
4. **Classification**: Dense layers output binary flare/no-flare predictions
5. **Visualization**: Generates attention maps showing what the model focuses on

**Key Features:**
- Self-attention mechanisms that focus on flare rise and decay phases
- Enhanced detection of low-energy flares (0.77 recall vs 0.70 for CNNs)
- Interpretable attention visualizations
- F1 score of 0.83 on ~53,000 labeled flare events

## Installation
```bash
pip install torch numpy matplotlib seaborn pandas scikit-learn pyyaml lightkurve
```

## Configuration

The model uses a YAML configuration file to control training behavior:

### Model Configuration
```yaml
model:
  input_dim: 3              # Number of input features per timestep
  d_model: 256              # Model dimensionality
  num_heads: 8              # Number of attention heads
  num_layers: 2             # Number of transformer encoder layers
  d_ff: 512                 # Feed-forward network dimensionality
  max_seq_len: 100          # Maximum sequence length
  dropout: 0.1              # Dropout rate

training:
  batch_size: 64            # Batch size for training
  num_epochs: 100           # Number of training epochs
  learning_rate: 0.0003     # Initial learning rate
  weight_decay: 0.00001     # Weight decay for regularization
  early_stopping_patience: 10  # Patience for early stopping
  seed: 42                  # Random seed for reproducibility

class_weights:
  flare: 1.4                # Weight for flare class
  non_flare: 0.7            # Weight for non-flare class

data:
  processed_dir: 'data/processed'  # Directory containing processed data
  num_workers: 4                   # Number of data loading workers
  train_split: 0.8                 # Training data split ratio
  val_split: 0.1                   # Validation data split ratio
  test_split: 0.1                  # Test data split ratio

output:
  results_dir: 'results'           # Directory to save results
  checkpoint_dir: 'checkpoints'    # Directory to save model checkpoints
  plots_dir: 'plots'               # Directory to save plots
  attention_maps_dir: 'attention_maps'  # Directory to save attention visualizations
```

## Quick Start

Create a `main.py` file to train your model:
```python
from src.training.training_main import train_model
from src.config.model_config import ModelConfig

def main():
    # Load configuration
    config = ModelConfig('src/config/config.yaml')
    
    # Train model
    model, history = train_model(config, device='cuda')
    
    print("Training complete!")

if __name__ == '__main__':
    main()
```

Your project directory should be organized as follows:
```
project-directory/
├── main.py
├── src/
│   ├── config/
│   │   └── config.yaml
│   ├── model/
│   ├── training/
│   └── visualization/
├── data/
│   └── processed/
```

Run your training:
```bash
python main.py
```

**Note**: Training takes approximately 14 hours on a single NVIDIA RTX 8000 GPU.

## Model Architecture

The transformer model consists of the following components:

### Architecture Details

- **Input Embedding**: Projects 3D input features (flux, gradient, quality) to d_model dimensions
- **Positional Encoding**: Sine/cosine functions encode temporal information
- **Transformer Encoder**: 2 layers with 8-head self-attention
- **Classification Head**: 
  - Flatten layer
  - Dense layer (512 units) + ReLU + Dropout
  - Dense layer (128 units) + ReLU + Dropout
  - Output layer (1 unit) + Sigmoid

### Mathematical Formulation

The self-attention mechanism follows:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Where Q, K, and V are linear projections of the input, and d_k is the key dimension.

## Performance

### Overall Metrics

| Metric    | Score | Comparison |
|-----------|-------|------------|
| F1 Score  | 0.83  | +0.01 vs CNN (0.82) |
| Precision | 0.81  | +0.02 vs CNN (0.79) |
| Recall    | 0.86  | +0.01 vs CNN (0.85) |
| AUC-ROC   | 0.93  | +0.02 vs CNN (0.91) |

### Low-Energy Flare Detection

The transformer shows significant improvement for challenging cases:

- **Low-energy flares**: 0.77 recall (vs 0.70 for CNN)
- **Multiple flares**: 0.85 F1 score (vs 0.77 for CNN)
- **Complex variability**: Maintains consistent performance across stellar types

## Output Files

After training, your results will be organized as follows:
```
results/
├── [timestamp]/
│   ├── checkpoints/
│   │   ├── best_model.pth           # Best model based on validation F1
│   │   └── checkpoint_epoch_*.pth   # Periodic checkpoints
│   ├── plots/
│   │   └── training_history.png     # Training and validation curves
│   ├── attention_maps/
│   │   ├── sample_*_lightcurve.png  # Light curve visualizations
│   │   └── sample_*_layer_*_attention.png  # Attention heatmaps
│   ├── config.yaml                   # Configuration used for this run
│   └── config.json                   # Training summary and final metrics
```

### Output Data Format

#### Training History

Training curves include:
- Training and validation loss
- Training and validation accuracy
- Validation precision and recall
- Validation F1 score

#### Attention Maps

Attention visualizations show:
- **Light curves**: Normalized flux over time for flare and non-flare examples
- **Attention heatmaps**: 8 attention heads per layer showing query-key relationships
- **Attention rollout**: Aggregated attention across all layers

#### Model Checkpoints

Checkpoint files contain:
- `model_state_dict`: Trained model weights
- `optimizer_state_dict`: Optimizer state for resuming training
- `epoch`: Training epoch number
- `val_metrics`: Validation metrics at checkpoint

## Repository Structure
```
src/
├── config/              # Configuration management
│   ├── config.yaml      # Default configuration
│   └── model_config.py  # Configuration validation
├── model/               # Transformer architecture
│   ├── transformer_model.py  # Main model definition
│   ├── attention.py          # Multi-head attention
│   ├── trainer.py            # Training procedures
│   └── loss.py               # Loss functions
├── training/            # Training scripts
│   ├── training_main.py      # Main training script
│   ├── training_plotter.py   # Training visualizations
│   └── utility.py            # Helper functions
└── visualization/       # Analysis and visualization
    ├── attention_visualizer.py  # Attention analysis
    ├── metrics_evaluation.py    # Performance metrics
    └── error_analysis.py        # Error categorization
```

## Contact

Isabella Longo - University of Colorado Boulder

---

**Thesis Committee**: Liz Bradley (Chair), Rachel Cox, Majid Zamani, David Wilson
