import lightkurve as lk
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path
from tqdm.auto import tqdm

class TrainingData:
    """
    """
    
    def __init__(self, cadence=120):
        self.cadence = cadence
        
        # Set up project directory structure
        self.project_dir = Path(__file__).resolve().parents[2]  # gets you to root directory
        self.processed_dir = self.project_dir / 'src' / 'data' / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Create separate directories for train and validation sets
        self.train_dir = self.processed_dir / 'train'
        self.val_dir = self.processed_dir / 'val'
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)

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

    def get_lightcurve_window(self, tic_id, sector, peak_time, window_size=100):
        """Extract a fixed-length window around a specific time in a lightcurve."""
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

    def process_and_save_data(self, flare_csv, sequence_length=100, train_split=0.8):
        """Process all lightcurves and save them to disk in HDF5 format."""
        # Load flare catalog
        data = pd.read_csv(self.project_dir / 'src' / 'data' / 'raw' / flare_csv)
        
        # Create HDF5 files for train and validation sets
        train_file = h5py.File(self.train_dir / 'flare_data.h5', 'w')
        val_file = h5py.File(self.val_dir / 'flare_data.h5', 'w')
        
        # Calculate split indices
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:int(train_split * n_samples)]
        val_idx = indices[int(train_split * n_samples):]
        
        # Create datasets in the HDF5 files
        train_flux = train_file.create_dataset('flux', 
                                             shape=(len(train_idx), sequence_length),
                                             dtype='float32')
        train_time = train_file.create_dataset('time', 
                                             shape=(len(train_idx), sequence_length),
                                             dtype='float32')
        train_labels = train_file.create_dataset('labels', 
                                               shape=(len(train_idx),),
                                               dtype='int8')
        train_tic = train_file.create_dataset('tic_id', 
                                            shape=(len(train_idx),),
                                            dtype='int64')
        
        val_flux = val_file.create_dataset('flux', 
                                         shape=(len(val_idx), sequence_length),
                                         dtype='float32')
        val_time = val_file.create_dataset('time', 
                                         shape=(len(val_idx), sequence_length),
                                         dtype='float32')
        val_labels = val_file.create_dataset('labels', 
                                           shape=(len(val_idx),),
                                           dtype='int8')
        val_tic = val_file.create_dataset('tic_id', 
                                        shape=(len(val_idx),),
                                        dtype='int64')
        
        # Process training data
        print("Processing training data...")
        for i, idx in enumerate(train_idx):
            row = data.iloc[idx]
            time, flux, success = self.get_lightcurve_window(
                row['TIC'],
                row['TESS Sector'],
                row['Flare peak time (BJD)'],
                sequence_length
            )
            
            if success:
                train_flux[i] = flux
                train_time[i] = time
                train_labels[i] = 1 if row['Number of fitted flarres'] >= 2 else 0
                train_tic[i] = row['TIC']
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(train_idx)} training samples")
        
        # Process validation data
        print("Processing validation data...")
        for i, idx in enumerate(val_idx):
            row = data.iloc[idx]
            time, flux, success = self.get_lightcurve_window(
                row['TIC'],
                row['TESS Sector'],
                row['Flare peak time (BJD)'],
                sequence_length
            )
            
            if success:
                val_flux[i] = flux
                val_time[i] = time
                val_labels[i] = 1 if row['Possible flare detection'] == 'Y' else 0
                val_tic[i] = row['TIC']
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(val_idx)} validation samples")
        
        # Save metadata
        metadata = {
            'sequence_length': sequence_length,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'creation_date': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata to JSON
        pd.Series(metadata).to_json(self.processed_dir / 'metadata.json')
        
        train_file.close()
        val_file.close()
        
        return self.processed_dir