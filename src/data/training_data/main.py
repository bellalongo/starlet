from pathlib import Path
from gen_training_data import TrainingData
from dataset import create_data_loaders
import argparse

def process_data(args):
    """Process raw flare data into training/validation sets"""
    processor = TrainingData(
        cadence=args.cadence,
        batch_size=args.batch_size
    )
    
    processor.process_and_save_data(
        flare_csv=args.input_csv,
        sequence_length=args.sequence_length,
        train_split=args.train_split,
        chunk_size=args.chunk_size
    )

def create_loaders(args):
    """Create data loaders from processed data"""
    # Convert processed_dir to an absolute path if it's relative
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        # Get the directory where the script is located
        script_dir = Path(__file__).parent.absolute()
        # Go up one level to the data directory, assuming training_data is under data/
        data_dir = script_dir.parent
        # Construct the full path
        processed_dir = data_dir / 'processed'
    
    print(f"Looking for data in: {processed_dir}")
    
    if not (processed_dir / 'train' / 'flare_data.h5').exists():
        raise FileNotFoundError(f"Training data file not found at {processed_dir / 'train' / 'flare_data.h5'}")
    
    train_loader, val_loader = create_data_loaders(
        processed_dir=processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    # train_loader, val_loader = create_data_loaders(
    #     processed_dir=Path(args.processed_dir),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers
    # )
    
    # Print sample batch info
    print("\nData loader information:")
    for batch_idx, (flux, label, multiple_flares) in enumerate(train_loader):
        print(f"Batch shape: {flux.shape}")  # Should be [batch_size, sequence_length, 1]
        print(f"Label shape: {label.shape}")  # Should be [batch_size, 1]
        print(f"Multiple flares shape: {multiple_flares.shape}")  # Should be [batch_size, sequence_length]
        print(f"Sample target values: {label[:5].numpy().flatten()}")  # Show first 5 labels
        break

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Process TESS flare data and create dataloaders')
    
    # Data processing arguments
    parser.add_argument('--input-csv', type=str, default='flares.csv',
                        help='Input CSV file containing flare data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory for processed data')
    parser.add_argument('--cadence', type=int, default=120,
                        help='Cadence in seconds (default: 120 for 2-minute cadence)')
    parser.add_argument('--sequence-length', type=int, default=100,
                        help='Length of each sequence window')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data to use for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Number of samples to process in each chunk')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Mode selection
    parser.add_argument('--mode', choices=['process', 'load', 'both'], default='both',
                        help='Mode to run: process data, load data, or both')
    
    args = parser.parse_args()
    
    if args.mode in ['process', 'both']:
        print("Processing raw data...")
        process_data(args)
    
    if args.mode in ['load', 'both']:
        print("\nCreating data loaders...")
        train_loader, val_loader = create_loaders(args)
        return train_loader, val_loader  # Return the loaders for use in training

if __name__ == "__main__":
    main()