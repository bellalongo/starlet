from pathlib import Path


class TrainingConfig:
    """
        Configuration class for training parameters.
        
        This class validates and stores all configuration parameters needed for
        training the transformer model, following the methodology outlined in the thesis.
        
        Attributes:
            processed_dir (Path): Directory containing processed data
            output_dir (Path): Directory to save results
            batch_size (int): Batch size for training
            num_workers (int): Number of workers for data loading
            d_model (int): Model dimensionality
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            d_ff (int): Feed-forward network dimensionality
            dropout (float): Dropout rate
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            num_epochs (int): Number of epochs to train
            seed (int): Random seed for reproducibility
            device (str): Device to use for training ('cuda' or 'cpu')
    """
    
    def __init__(self, args):
        """
            Initializes the training configuration from command line arguments.
            
            Arguments:
                args: Parsed command line arguments
                
            Raises:
                ValueError: If any parameter has an invalid value
                TypeError: If any parameter has an incorrect type
        """
        self.processed_dir = Path(args.processed_dir)
        self.output_dir = Path(args.output_dir)
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        # Model architecture parameters
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        
        # Training parameters
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.seed = args.seed
        self.device = args.device
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """
            Validates all configuration parameters.
            
            Raises:
                ValueError: If any parameter has an invalid value
                TypeError: If any parameter has an incorrect type
        """
        # Check types
        if not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be of type 'int'")
        
        if not isinstance(self.num_workers, int):
            raise TypeError("num_workers must be of type 'int'")
        
        if not isinstance(self.d_model, int):
            raise TypeError("d_model must be of type 'int'")
        
        if not isinstance(self.num_heads, int):
            raise TypeError("num_heads must be of type 'int'")
        
        if not isinstance(self.num_layers, int):
            raise TypeError("num_layers must be of type 'int'")
        
        # Check values
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        
        if not self.processed_dir.exists():
            raise ValueError(f"Processed data directory does not exist: {self.processed_dir}")