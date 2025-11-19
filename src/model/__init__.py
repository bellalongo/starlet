"""
    Transformer-based stellar flare detection model.
"""

from .transformer_model import (
    FlareClassifier,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_model
)

from .attention import MultiHeadAttention

from .trainer import Trainer, train_flare_detection_model

from .loss import WeightedBCELoss, get_class_weights

__all__ = [
    # Main model
    'FlareClassifier',
    'create_model',
    
    # Architecture components
    'PositionalEncoding',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'MultiHeadAttention',
    
    # Training
    'Trainer',
    'train_flare_detection_model',
    
    # Loss functions
    'WeightedBCELoss',
    'get_class_weights',
]