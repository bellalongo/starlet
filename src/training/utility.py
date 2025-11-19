import os
import numpy as np
import random
import torch


def set_seed(seed):
    """
        Set random seed for reproducibility across all libraries.
        
        Arguments:
            seed (int): Random seed value
            
        Returns:
            None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_directories(output_dir, timestamp):
    """
        Creates necessary output directories for training results.
        
        Arguments:
            output_dir (Path): Base output directory
            timestamp (str): Timestamp string for this training run
            
        Returns:
            dict: Dictionary containing paths to created directories
    """
    run_dir = output_dir / timestamp
    
    directories = {
        'run': run_dir,
        'checkpoints': run_dir / 'checkpoints',
        'plots': run_dir / 'plots',
        'attention_maps': run_dir / 'attention_maps'
    }
    
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories
