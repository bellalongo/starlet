from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import lightkurve as lk
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil
import time
import os

class TrainingData:
    """
    Process TESS lightcurves into features for the transformer model.
    Handles multiple flares within single lightcurves and creates windowed data
    for training the transformer.
    """
    
    def __init__(self, cadence=120, batch_size=1000):
        self.cadence = cadence
        self.batch_size = batch_size
        self.n_workers = self._get_optimal_workers()
        
        # Set up project directory structure
        self.project_dir = Path(__file__).resolve().parents[2]
        self.processed_dir = self.project_dir / 'data' / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        self.train_dir = self.processed_dir / 'train'
        self.val_dir = self.processed_dir / 'val'
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)
        
        self.state_file = self.processed_dir / 'processing_state.json'
        self.checkpoint_dir = self.processed_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"Initialized with {self.n_workers} workers and batch size {batch_size}")

    def _get_optimal_workers(self):
        """Determine optimal number of workers based on system resources"""
        cpu_count = multiprocessing.cpu_count()
        available_memory = psutil.virtual_memory().available
        suggested_workers = max(2, int(cpu_count * 0.75))
        max_workers_by_memory = max(2, int(available_memory / (2 * 1024**3)))
        return min(suggested_workers, max_workers_by_memory)

    def check_nearby_flares(self, current_peak_time, all_peak_times, window_duration):
        """Check if there are other flares within the window duration."""
        half_window = window_duration / 2
        window_start = current_peak_time - half_window
        window_end = current_peak_time + half_window
        
        nearby_flares = [t for t in all_peak_times 
                        if window_start <= t <= window_end 
                        and t != current_peak_time]
        
        return len(nearby_flares) > 0, nearby_flares

    def append_lightcurves(self, result, result_exposures):
        """Append multiple lightcurves of the same cadence."""
        all_lightcurves = []

        # Get the data whose exposure is the desired cadence
        for i, exposure in enumerate(result_exposures):
            if exposure.value == self.cadence:
                lightcurve = result[i].download().remove_nans().remove_outliers().normalize() - 1
                all_lightcurves.append(lightcurve)
        
        if not all_lightcurves:
            return None
            
        # Combine all lightcurves
        combined_lightcurve = all_lightcurves[0]
        for lc in all_lightcurves[1:]:
            combined_lightcurve = combined_lightcurve.append(lc)
        
        return combined_lightcurve

    def get_lightcurve_window(self, tic_id, sector, peak_time, all_peak_times, window_size=100):
        """Extract a fixed-length window around a specific time in a lightcurve."""
        try:
            # Search for the light curve
            result = lk.search_lightcurve(f'TIC {tic_id}', sector=sector)
            if len(result) == 0:
                return None, None, False, None
                
            # Get exposure and the appended lightcurve
            result_exposures = result.exptime
            lightcurve = self.append_lightcurves(result, result_exposures)
            
            if lightcurve is None:
                return None, None, False, None
            
            # Calculate time window duration in days
            time_array = lightcurve.time.value
            cadence_days = np.median(np.diff(time_array))
            window_duration = cadence_days * window_size
            
            # Check for multiple flares
            has_multiple, nearby_flares = self.check_nearby_flares(
                peak_time, all_peak_times, window_duration)
            
            # Find the closest time index to peak_time
            peak_idx = np.argmin(np.abs(time_array - peak_time))
            
            # Calculate window boundaries with random offset for data augmentation
            offset = np.random.randint(-20, 20)  # Random offset to avoid centering bias
            start_idx = max(0, peak_idx - window_size//2 + offset)
            end_idx = min(len(time_array), start_idx + window_size)
            
            # Extract time and flux arrays
            time = np.array(time_array[start_idx:end_idx])
            flux = np.array(lightcurve.flux.value[start_idx:end_idx])
            
            # Create a mask indicating where other flares occur in the window
            flare_mask = np.zeros_like(time, dtype=bool)
            if has_multiple:
                for flare_time in nearby_flares:
                    flare_idx = np.argmin(np.abs(time - flare_time))
                    flare_mask[flare_idx] = True
            
            # Handle cases where we don't have enough points
            if len(time) < window_size:
                pad_length = window_size - len(time)
                time = np.pad(time, (0, pad_length), 'constant', constant_values=np.nan)
                flux = np.pad(flux, (0, pad_length), 'constant', constant_values=np.nan)
                flare_mask = np.pad(flare_mask, (0, pad_length), 'constant', constant_values=False)
            
            # Replace NaNs with zeros for the transformer
            flux = np.nan_to_num(flux, nan=0.0)
            
            # Validate the window
            if not self.validate_window(flux, time, window_size):
                return None, None, False, None
            
            return time, flux, True, flare_mask
            
        except Exception as e:
            print(f"Error processing TIC {tic_id}: {str(e)}")
            return None, None, False, None

    def validate_window(self, flux, time, window_size):
        """Validate the extracted window meets quality criteria."""
        if len(flux) != window_size:
            return False
        if np.sum(np.isnan(flux)) > window_size * 0.1:  # Max 10% NaN
            return False
        if np.std(flux) <= 0:  # Check for flat lightcurves
            return False
        return True

    def process_batch(self, batch_data):
        """Process a batch of lightcurve data."""
        results = []
        
        # Group the batch by TIC ID and sector to handle multiple flares
        grouped = batch_data.groupby(['TIC', 'TESS Sector'])
        
        # Add progress bar for stars in the batch
        for (tic_id, sector), star_data in tqdm(grouped, desc=f"Processing TIC stars", leave=False):
            # Get all flare peak times for this star
            peak_times = star_data['Flare peak time (BJD)'].values
            
            # Process each flare with progress bar
            for _, row in tqdm(star_data.iterrows(), desc=f"Processing flares for TIC {tic_id}", leave=False):
                try:
                    time, flux, success, flare_mask = self.get_lightcurve_window(
                        row['TIC'],
                        sector,
                        row['Flare peak time (BJD)'],
                        peak_times,
                        self.sequence_length
                    )
                    
                    if success:
                        label = 1 if row['Number of fitted flare profiles'] >= 2 else 0
                        results.append({
                            'index': row.name,
                            'flux': flux,
                            'time': time,
                            'label': label,
                            'tic_id': row['TIC'],
                            'multiple_flares': flare_mask,
                            'success': True
                        })
                except Exception as e:
                    print(f"Error processing TIC {row['TIC']}: {str(e)}")
                    continue
        
        return results

    def verify_saved_data(self, h5_file, batch_results, start_idx):
        """Verify that data was correctly saved to HDF5"""
        verification_passed = True
        for result in batch_results:
            if result['success']:
                idx = result['index']
                saved_flux = h5_file['flux'][idx]
                saved_time = h5_file['time'][idx]
                
                if not np.allclose(saved_flux, result['flux'], equal_nan=True) or \
                   not np.allclose(saved_time, result['time'], equal_nan=True):
                    verification_passed = False
                    print(f"Verification failed for index {idx}")
                    break
        
        return verification_passed
    
    def save_checkpoint(self, batch_num, results, state):
        """Save checkpoint data"""
        checkpoint_file = self.checkpoint_dir / f'checkpoint_batch_{batch_num}.json'
        checkpoint_data = {
            'batch_num': batch_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_processed': len(results),
            'state': state,
            'progress': {
                'training_complete': len(state['processed_train']),
                'validation_complete': len(state['processed_val']),
                'total_samples': len(state['train_idx']) + len(state['val_idx'])
            }
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"\nSaved checkpoint to {checkpoint_file}")
        print(f"Progress: {len(state['processed_train'])} training, {len(state['processed_val'])} validation samples")

    def process_and_save_data(self, flare_csv, sequence_length=100, train_split=0.8, chunk_size=20):
        """
        Process lightcurves with parallel processing and checkpointing.
        
        Args:
            flare_csv (str): Name of CSV file containing flare data
            sequence_length (int): Length of each sequence window
            train_split (float): Fraction of data to use for training
            chunk_size (int): Number of samples to process in each chunk
        """
        self.sequence_length = sequence_length
        
        # Load flare catalog
        print("\nLoading flare catalog...")
        data = pd.read_csv(self.project_dir / 'data' / 'raw' / flare_csv)
        print(f"Found {len(data)} total samples")
        
        # Load or create processing state
        state = self.load_or_create_state(len(data), train_split)
        
        # Split data into train and validation chunks
        train_data = data.iloc[state['train_idx']]
        val_data = data.iloc[state['val_idx']]
        
        # Create overall progress bar
        total_samples = len(data)
        processed_samples = len(state['processed_train']) + len(state['processed_val'])
        
        with tqdm(total=total_samples, initial=processed_samples, desc="Overall Progress") as pbar:
            with h5py.File(self.train_dir / 'flare_data.h5', 'a') as train_file, \
                h5py.File(self.val_dir / 'flare_data.h5', 'a') as val_file:
                
                # Initialize datasets
                train_datasets = self.initialize_or_get_datasets(
                    train_file, len(state['train_idx']), sequence_length)
                val_datasets = self.initialize_or_get_datasets(
                    val_file, len(state['val_idx']), sequence_length)
                
                # Process training data
                remaining_train = train_data[~train_data.index.isin(state['processed_train'])]
                chunks = [remaining_train[i:i + chunk_size] 
                        for i in range(0, len(remaining_train), chunk_size)]
                
                if len(remaining_train) > 0:
                    print(f"\nProcessing {len(remaining_train)} remaining training samples in {len(chunks)} chunks...")
                
                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        futures = [executor.submit(self.process_batch, chunk) for chunk in chunks]
                        
                        for batch_num, future in enumerate(tqdm(futures, desc="Training Batches", leave=False)):
                            try:
                                results = future.result()
                                for result in results:
                                    if result['success']:
                                        original_idx = result['index']
                                        # Find the position of this index in the train_idx list
                                        if original_idx in state['train_idx']:
                                            # Get the position within the training array
                                            array_idx = state['train_idx'].index(original_idx)
                                            train_datasets['flux'][array_idx] = result['flux']
                                            train_datasets['time'][array_idx] = result['time']
                                            train_datasets['labels'][array_idx] = result['label']
                                            train_datasets['tic_id'][array_idx] = result['tic_id']
                                            train_datasets['multiple_flares'][array_idx] = result['multiple_flares']
                                            state['processed_train'].append(original_idx)
                                            pbar.update(1)
                                
                                # Save checkpoint and state
                                if len(results) > 0:
                                    self.save_checkpoint(batch_num, results, state)
                                    self.save_state(state)
                                    
                            except Exception as e:
                                print(f"\nError processing batch {batch_num}: {str(e)}")
                                continue
                
                # Process validation data
                remaining_val = val_data[~val_data.index.isin(state['processed_val'])]
                chunks = [remaining_val[i:i + chunk_size] 
                        for i in range(0, len(remaining_val), chunk_size)]
                
                if len(remaining_val) > 0:
                    print(f"\nProcessing {len(remaining_val)} remaining validation samples in {len(chunks)} chunks...")
                
                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        futures = [executor.submit(self.process_batch, chunk) for chunk in chunks]
                        
                        for batch_num, future in enumerate(tqdm(futures, desc="Validation Batches", leave=False)):
                            try:
                                results = future.result()
                                for result in results:
                                    if result['success']:
                                        original_idx = result['index']
                                        # Find the position of this index in the val_idx list
                                        if original_idx in state['val_idx']:
                                            # Get the position within the validation array
                                            array_idx = state['val_idx'].index(original_idx)
                                            val_datasets['flux'][array_idx] = result['flux']
                                            val_datasets['time'][array_idx] = result['time']
                                            val_datasets['labels'][array_idx] = result['label']
                                            val_datasets['tic_id'][array_idx] = result['tic_id']
                                            val_datasets['multiple_flares'][array_idx] = result['multiple_flares']
                                            state['processed_val'].append(original_idx)
                                            pbar.update(1)
                                
                                # Save checkpoint and state
                                if len(results) > 0:
                                    self.save_checkpoint(batch_num + len(chunks), results, state)
                                    self.save_state(state)
                                    
                            except Exception as e:
                                print(f"\nError processing validation batch {batch_num}: {str(e)}")
                                continue

            # Save final metadata
            metadata = {
                'sequence_length': sequence_length,
                'train_samples': len(state['train_idx']),
                'val_samples': len(state['val_idx']),
                'creation_date': pd.Timestamp.now().isoformat(),
                'completed': True,
                'stats': {
                    'processed_train': len(state['processed_train']),
                    'processed_val': len(state['processed_val']),
                    'total_processed': len(state['processed_train']) + len(state['processed_val']),
                    'success_rate': (len(state['processed_train']) + len(state['processed_val'])) / total_samples
                }
            }
            
            metadata_file = self.processed_dir / 'metadata.json'
            pd.Series(metadata).to_json(metadata_file)
            print(f"\nProcessing completed!")
            print(f"Saved metadata to {metadata_file}")
            print(f"\nFinal Statistics:")
            print(f"  Training samples processed: {len(state['processed_train'])}/{len(state['train_idx'])}")
            print(f"  Validation samples processed: {len(state['processed_val'])}/{len(state['val_idx'])}")
            print(f"  Overall success rate: {metadata['stats']['success_rate']*100:.2f}%")

    def load_or_create_state(self, n_samples, train_split=0.8):
        """Initialize or load existing processing state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            print("Resuming from previous state...")
        else:
            # Create new train/val split
            indices = np.random.permutation(n_samples)
            train_idx = indices[:int(train_split * n_samples)].tolist()
            val_idx = indices[int(train_split * n_samples):].tolist()
            
            state = {
                'train_idx': train_idx,
                'val_idx': val_idx,
                'processed_train': [],
                'processed_val': [],
                'train_split': train_split
            }
            self.save_state(state)
            
        return state

    def save_state(self, state):
        """Save current processing state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
        print(f"\nSaved state to {self.state_file}")
        print(f"Progress: {len(state['processed_train'])} training samples, {len(state['processed_val'])} validation samples processed")

    def initialize_or_get_datasets(self, file, n_samples, sequence_length):
        """Initialize datasets if they don't exist, or return existing ones."""
        datasets = {}
        
        # Define dataset specifications
        specs = {
            'flux': ('float32', (n_samples, sequence_length)),
            'time': ('float32', (n_samples, sequence_length)),
            'labels': ('int8', (n_samples,)),
            'tic_id': ('int64', (n_samples,)),
            'multiple_flares': ('bool', (n_samples, sequence_length))  # Added for multiple flares
        }
        
        for name, (dtype, shape) in specs.items():
            if name in file:
                datasets[name] = file[name]
            else:
                datasets[name] = file.create_dataset(name, shape=shape, dtype=dtype)
                
        return datasets
