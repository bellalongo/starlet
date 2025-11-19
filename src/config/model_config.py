"""
    Configuration management for the transformer model.
"""

import yaml
from pathlib import Path


# ------------------------------
# ModelConfig Class
# ------------------------------
class ModelConfig:
    """
        Configuration class for model architecture and training parameters.
        
        This class loads and validates configuration from YAML files,
        following the validation pattern of InputCheck in spec2flux.
        
        Attributes:
            # Model architecture
            input_dim (int): Dimensionality of input features
            d_model (int): Model dimensionality
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            d_ff (int): Feed-forward network dimensionality
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout rate
            
            # Training parameters
            batch_size (int): Batch size for training
            num_epochs (int): Number of epochs to train
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            early_stopping_patience (int): Patience for early stopping
            seed (int): Random seed for reproducibility
            
            # Class weights
            flare_weight (float): Weight for flare class
            non_flare_weight (float): Weight for non-flare class
            
            # Data configuration
            processed_dir (Path): Directory containing processed data
            num_workers (int): Number of workers for data loading
            train_split (float): Training data split ratio
            val_split (float): Validation data split ratio
            test_split (float): Test data split ratio
            
            # Output configuration
            results_dir (Path): Directory to save results
            checkpoint_dir (Path): Directory to save checkpoints
            plots_dir (Path): Directory to save plots
            attention_maps_dir (Path): Directory to save attention maps
    """
    
    def __init__(self, config_path=None):
        """
            Initializes configuration from YAML file.
            
            Arguments:
                config_path (str or Path, optional): Path to configuration file.
                    If None, uses default_config.yaml in the config directory.
                    
            Raises:
                FileNotFoundError: If config file doesn't exist
                ValueError: If any parameter has an invalid value
                TypeError: If any parameter has an incorrect type
        """
        # Use default config if none provided
        if config_path is None:
            config_path = Path(__file__).parent / 'default_config.yaml'
        else:
            config_path = Path(config_path)
        
        # Load configuration
        self._load_config(config_path)
        
        # Validate configuration
        self._validate_config()
    
    # ------------------------------
    # Private Helper Methods
    # ------------------------------
    def _load_config(self, config_path):
        """
            Loads configuration from YAML file.
            
            Arguments:
                config_path (Path): Path to configuration file
                
            Returns:
                None
                
            Raises:
                FileNotFoundError: If config file doesn't exist
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Model architecture
        model_config = config['model']
        self.input_dim = model_config['input_dim']
        self.d_model = model_config['d_model']
        self.num_heads = model_config['num_heads']
        self.num_layers = model_config['num_layers']
        self.d_ff = model_config['d_ff']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']
        
        # Training parameters
        training_config = config['training']
        self.batch_size = training_config['batch_size']
        self.num_epochs = training_config['num_epochs']
        self.learning_rate = training_config['learning_rate']
        self.weight_decay = training_config['weight_decay']
        self.early_stopping_patience = training_config['early_stopping_patience']
        self.seed = training_config['seed']
        
        # Class weights
        weights_config = config['class_weights']
        self.flare_weight = weights_config['flare']
        self.non_flare_weight = weights_config['non_flare']
        
        # Data configuration
        data_config = config['data']
        self.processed_dir = Path(data_config['processed_dir'])
        self.num_workers = data_config['num_workers']
        self.train_split = data_config['train_split']
        self.val_split = data_config['val_split']
        self.test_split = data_config['test_split']
        
        # Output configuration
        output_config = config['output']
        self.results_dir = Path(output_config['results_dir'])
        self.checkpoint_dir = Path(output_config['checkpoint_dir'])
        self.plots_dir = Path(output_config['plots_dir'])
        self.attention_maps_dir = Path(output_config['attention_maps_dir'])
    
    def _validate_config(self):
        """
            Validates all configuration parameters.
            
            Arguments:
                None
                
            Returns:
                None
                
            Raises:
                ValueError: If any parameter has an invalid value
                TypeError: If any parameter has an incorrect type
        """
        # Validate model architecture parameters
        self._validate_model_params()
        
        # Validate training parameters
        self._validate_training_params()
        
        # Validate class weights
        self._validate_class_weights()
        
        # Validate data configuration
        self._validate_data_params()
    
    def _validate_model_params(self):
        """
            Validates model architecture parameters.
            
            Arguments:
                None
                
            Returns:
                None
                
            Raises:
                TypeError: If any parameter has an incorrect type
                ValueError: If any parameter has an invalid value
        """
        # Check types
        if not isinstance(self.input_dim, int):
            raise TypeError("input_dim must be of type 'int'")
        
        if not isinstance(self.d_model, int):
            raise TypeError("d_model must be of type 'int'")
        
        if not isinstance(self.num_heads, int):
            raise TypeError("num_heads must be of type 'int'")
        
        if not isinstance(self.num_layers, int):
            raise TypeError("num_layers must be of type 'int'")
        
        if not isinstance(self.d_ff, int):
            raise TypeError("d_ff must be of type 'int'")
        
        if not isinstance(self.max_seq_len, int):
            raise TypeError("max_seq_len must be of type 'int'")
        
        if not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be of type 'int' or 'float'")
        
        # Check values
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
    
    def _validate_training_params(self):
        """
            Validates training parameters.
            
            Arguments:
                None
                
            Returns:
                None
                
            Raises:
                TypeError: If any parameter has an incorrect type
                ValueError: If any parameter has an invalid value
        """
        # Check types
        if not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be of type 'int'")
        
        if not isinstance(self.num_epochs, int):
            raise TypeError("num_epochs must be of type 'int'")
        
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate must be of type 'int' or 'float'")
        
        if not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be of type 'int' or 'float'")
        
        if not isinstance(self.early_stopping_patience, int):
            raise TypeError("early_stopping_patience must be of type 'int'")
        
        if not isinstance(self.seed, int):
            raise TypeError("seed must be of type 'int'")
        
        # Check values
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
    
    def _validate_class_weights(self):
        """
            Validates class weight parameters.
            
            Arguments:
                None
                
            Returns:
                None
                
            Raises:
                TypeError: If any parameter has an incorrect type
                ValueError: If any parameter has an invalid value
        """
        # Check types
        if not isinstance(self.flare_weight, (int, float)):
            raise TypeError("flare_weight must be of type 'int' or 'float'")
        
        if not isinstance(self.non_flare_weight, (int, float)):
            raise TypeError("non_flare_weight must be of type 'int' or 'float'")
        
        # Check values
        if self.flare_weight <= 0:
            raise ValueError("flare_weight must be positive")
        
        if self.non_flare_weight <= 0:
            raise ValueError("non_flare_weight must be positive")
    
    def _validate_data_params(self):
        """
            Validates data configuration parameters.
            
            Arguments:
                None
                
            Returns:
                None
                
            Raises:
                TypeError: If any parameter has an incorrect type
                ValueError: If any parameter has an invalid value
        """
        # Check types
        if not isinstance(self.num_workers, int):
            raise TypeError("num_workers must be of type 'int'")
        
        if not isinstance(self.train_split, (int, float)):
            raise TypeError("train_split must be of type 'int' or 'float'")
        
        if not isinstance(self.val_split, (int, float)):
            raise TypeError("val_split must be of type 'int' or 'float'")
        
        if not isinstance(self.test_split, (int, float)):
            raise TypeError("test_split must be of type 'int' or 'float'")
        
        # Check values
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        
        if self.train_split <= 0 or self.train_split >= 1:
            raise ValueError("train_split must be between 0 and 1")
        
        if self.val_split <= 0 or self.val_split >= 1:
            raise ValueError("val_split must be between 0 and 1")
        
        if self.test_split <= 0 or self.test_split >= 1:
            raise ValueError("test_split must be between 0 and 1")
        
        # Check that splits sum to 1
        split_sum = self.train_split + self.val_split + self.test_split
        if not abs(split_sum - 1.0) < 1e-6:
            raise ValueError(
                f"train_split, val_split, and test_split must sum to 1.0, "
                f"got {split_sum}"
            )
    
    # ------------------------------
    # Public Methods
    # ------------------------------
    def to_dict(self):
        """
            Converts configuration to dictionary format.
            
            Arguments:
                None
                
            Returns:
                dict: Configuration as dictionary
        """
        return {
            'model': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout
            },
            'training': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'early_stopping_patience': self.early_stopping_patience,
                'seed': self.seed
            },
            'class_weights': {
                'flare': self.flare_weight,
                'non_flare': self.non_flare_weight
            },
            'data': {
                'processed_dir': str(self.processed_dir),
                'num_workers': self.num_workers,
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split
            },
            'output': {
                'results_dir': str(self.results_dir),
                'checkpoint_dir': str(self.checkpoint_dir),
                'plots_dir': str(self.plots_dir),
                'attention_maps_dir': str(self.attention_maps_dir)
            }
        }
    
    def save(self, save_path):
        """
            Saves configuration to YAML file.
            
            Arguments:
                save_path (str or Path): Path to save configuration
                
            Returns:
                None
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)