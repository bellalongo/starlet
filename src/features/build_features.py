import lightkurve as lk
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import psutil
import os

class LightcurveProcessor:
    def __init__(self, cadence=120, batch_size=1000):
        self.cadence = cadence
        self.batch_size = batch_size
        self.n_workers = self._get_optimal_workers()
        
        # Set up project directory structure
        self.project_dir = Path(__file__).resolve().parents[2]
        self.processed_dir = self.project_dir / 'src' / 'data' / 'processed'
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
        
        # Use 75% of available CPU cores, but at least 2
        suggested_workers = max(2, int(cpu_count * 0.75))
        
        # Ensure we don't use too much memory (assume 2GB per worker)
        max_workers_by_memory = max(2, int(available_memory / (2 * 1024**3)))
        
        return min(suggested_workers, max_workers_by_memory)

    def process_chunk(self, chunk_data):
        """Process a chunk of lightcurves"""
        results = []
        for _, row in chunk_data.iterrows():
            try:
                time, flux, success = self.get_lightcurve_window(
                    row['TIC'],
                    row['TESS Sector'],
                    row['Flare peak time (BJD)'],
                    self.sequence_length
                )
                
                if success:
                    results.append({
                        'index': row.name,
                        'flux': flux,
                        'time': time,
                        'label': 1 if row['Possible flare detection'] == 'Y' else 0,
                        'tic_id': row['TIC'],
                        'success': True
                    })
            except Exception as e:
                print(f"Error processing TIC {row['TIC']}: {str(e)}")
                continue
                
        return results

    def append_lightcurves(self, result, result_exposures):
        """
        Append multiple lightcurves of the same cadence.
        
        Parameters:
            result: Lightkurve search result
            result_exposures: Exposure times for each lightcurve
            
        Returns:
            combined_lightcurve: Single lightcurve combining all matching cadence data
        """
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

    def get_lightcurve_window(self, tic_id, sector, peak_time, window_size=100):
        """
        Extract a fixed-length window around a specific time in a lightcurve.
        
        Parameters:
            tic_id: TIC ID of the star
            sector: TESS sector
            peak_time: Time of interest (e.g., flare peak) in BJD
            window_size: Number of timesteps to include
            
        Returns:
            time: Array of timestamps
            flux: Array of normalized flux values
            success: Boolean indicating if retrieval was successful
        """
        try:
            # Search for the light curve
            result = lk.search_lightcurve(f'TIC {tic_id}', sector=sector)
            if len(result) == 0:
                return None, None, False
                
            # Get exposure and the appended lightcurve
            result_exposures = result.exptime
            lightcurve = self.append_lightcurves(result, result_exposures)
            
            if lightcurve is None:
                return None, None, False
            
            # Find the closest time index to peak_time
            peak_idx = np.argmin(np.abs(lightcurve.time.value - peak_time))
            
            # Calculate window boundaries with random offset for data augmentation
            offset = np.random.randint(-20, 20)  # Random offset to avoid centering bias
            start_idx = max(0, peak_idx - window_size//2 + offset)
            end_idx = min(len(lightcurve.time), start_idx + window_size)
            
            # Extract time and flux arrays
            time = np.array(lightcurve.time.value[start_idx:end_idx])
            flux = np.array(lightcurve.flux.value[start_idx:end_idx])
            
            # Handle cases where we don't have enough points
            if len(time) < window_size:
                pad_length = window_size - len(time)
                time = np.pad(time, (0, pad_length), 'constant', constant_values=np.nan)
                flux = np.pad(flux, (0, pad_length), 'constant', constant_values=np.nan)
            
            # Replace NaNs with zeros for the transformer
            flux = np.nan_to_num(flux, nan=0.0)
            
            return time, flux, True
            
        except Exception as e:
            print(f"Error processing TIC {tic_id}: {str(e)}")
            return None, None, False

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

    def process_and_save_data(self, flare_csv, sequence_length=100, train_split=0.8, chunk_size=20):
        """Process lightcurves in parallel using multiple processes"""
        self.sequence_length = sequence_length
        
        # Load flare catalog
        data = pd.read_csv(self.project_dir / 'src' / 'data' / 'raw' / flare_csv)
        
        # Load or create processing state
        state = self.load_or_create_state(len(data), train_split)
        
        # Split data into train and validation chunks
        train_data = data.iloc[state['train_idx']]
        val_data = data.iloc[state['val_idx']]
        
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
            
            print(f"Processing {len(remaining_train)} remaining training samples in {len(chunks)} chunks...")
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
                
                for future in tqdm(futures, desc="Processing training chunks"):
                    results = future.result()
                    for result in results:
                        if result['success']:
                            idx = result['index']
                            train_datasets['flux'][idx] = result['flux']
                            train_datasets['time'][idx] = result['time']
                            train_datasets['labels'][idx] = result['label']
                            train_datasets['tic_id'][idx] = result['tic_id']
                            state['processed_train'].append(idx)
                    
                    # Save state periodically
                    if len(results) > 0:
                        self.save_state(state)
            
            # Process validation data
            remaining_val = val_data[~val_data.index.isin(state['processed_val'])]
            chunks = [remaining_val[i:i + chunk_size] 
                     for i in range(0, len(remaining_val), chunk_size)]
            
            print(f"Processing {len(remaining_val)} remaining validation samples in {len(chunks)} chunks...")
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
                
                for future in tqdm(futures, desc="Processing validation chunks"):
                    results = future.result()
                    for result in results:
                        if result['success']:
                            idx = result['index']
                            val_datasets['flux'][idx] = result['flux']
                            val_datasets['time'][idx] = result['time']
                            val_datasets['labels'][idx] = result['label']
                            val_datasets['tic_id'][idx] = result['tic_id']
                            state['processed_val'].append(idx)
                    
                    # Save state periodically
                    if len(results) > 0:
                        self.save_state(state)

        # Save final metadata
        metadata = {
            'sequence_length': sequence_length,
            'train_samples': len(state['train_idx']),
            'val_samples': len(state['val_idx']),
            'creation_date': pd.Timestamp.now().isoformat(),
            'completed': True
        }
        
        pd.Series(metadata).to_json(self.processed_dir / 'metadata.json')
        print("Processing completed successfully!")

    def initialize_or_get_datasets(self, file, n_samples, sequence_length):
        """Initialize datasets if they don't exist, or return existing ones."""
        datasets = {}
        
        # Define dataset specifications
        specs = {
            'flux': ('float32', (n_samples, sequence_length)),
            'time': ('float32', (n_samples, sequence_length)),
            'labels': ('int8', (n_samples,)),
            'tic_id': ('int64', (n_samples,))
        }
        
        for name, (dtype, shape) in specs.items():
            if name in file:
                datasets[name] = file[name]
            else:
                datasets[name] = file.create_dataset(name, shape=shape, dtype=dtype)
                
        return datasets