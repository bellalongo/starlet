from dataset import *

# Example usage
if __name__ == "__main__":
    # Test the dataset and dataloaders
    # This assumes you've already run build_features.py
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / 'src' / 'data' / 'processed'
    
    train_loader, val_loader = create_data_loaders(processed_dir)
    
    # Print sample batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch shape: {data.shape}")  # Should be [batch_size, sequence_length, 1]
        print(f"Target shape: {target.shape}")  # Should be [batch_size, 1]
        print(f"Sample target values: {target[:5]}")  # Show first 5 labels
        break