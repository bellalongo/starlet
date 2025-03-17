from build_features import *

if __name__ == "__main__":
    # Initialize processor
    processor = LightcurveProcessor()
    
    # Process data
    processed_dir = processor.process_and_save_data('flares.csv', sequence_length=100)
    print(f"Data processed and saved to {processed_dir}")

    