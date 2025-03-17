from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class FlareDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        Initialize the FlareDataset.
        
        Args:
            data_dir (Path): Directory containing the processed data
            mode (str): Either 'train' or 'val'
        """
        self.data_path = data_dir / mode / 'flare_data.h5'
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Just store the path and check the dataset size
        with h5py.File(self.data_path, 'r') as f:
            self.length = len(f['labels'])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Open the file for each access
        with h5py.File(self.data_path, 'r') as f:
            # Get flux data and reshape for model input
            flux = torch.FloatTensor(f['flux'][idx]).unsqueeze(-1)  # Add channel dimension
            label = torch.FloatTensor([f['labels'][idx]])
            
            # Check if multiple_flares exists in the dataset
            if 'multiple_flares' in f:
                multiple_flares = torch.BoolTensor(f['multiple_flares'][idx])
            else:
                # Create a dummy array if it doesn't exist
                multiple_flares = torch.zeros(flux.shape[0], dtype=torch.bool)
                
        return flux, label, multiple_flares

def create_data_loaders(processed_dir, batch_size=32, num_workers=4):
    """
    Create DataLoader objects for training and validation.
    
    Args:
        processed_dir (Path): Directory containing processed data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = FlareDataset(processed_dir, mode='train')
    val_dataset = FlareDataset(processed_dir, mode='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader