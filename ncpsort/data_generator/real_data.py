import numpy as np
import torch
from torch.utils.data import Dataset

from ncpsort.utils.data_readers import load_waveform_by_spike_time

class SpikeRawDatasetByChannel(Dataset):
    """Create a Pytorch dataset of spike waveforms from raw recordings 

    Loads selected spikes in selected channels with fixed time window from raw voltage file

    Attributes:
        waveforms: A torch array of shape (n_spikes, n_channels, n_timesteps)
        unit_assignments: A numpy array of shape (n_spikes,) of unit assignments (cluster IDs) for the spikes
        spike_time: A numpy array of shape (n_spikes,) of the original spike time
        n_units: The number of ground truth units (clusters)
    """
    def __init__(self, voltage_file, channels, spike_time_labels, total_channels, 
                n_timesteps=None, time_offset=-30, start_idx=0, end_idx=-1, verbose=True):
        """Load the dataset and create torch array 
        Args:
            voltage_file: a ".bin" file of the recording from all channels
            channels: a list of channels to load
            spike_time_labels: (n_spikes, 2) each row [time, spike_label]
            total_channels: total number of channels (needed for loading the correct channels)
            n_timesteps: size of time window to keep 
            time_offset: peak offset
            start_idx, end_idx: trim dataset
        """
        print("loading spike data from channels {} ".format(channels))
        self.voltage_file = voltage_file
        self.total_channels = total_channels
        self.spike_time = spike_time_labels[start_idx:end_idx, 0]
        self.template_assignments = spike_time_labels[start_idx:end_idx, 1]
        self.channels = channels
        self.time_offset = time_offset

        self.waveforms = load_waveform_by_spike_time(
            voltage_file=self.voltage_file, spike_times=self.spike_time, 
            time_offset=self.time_offset, load_channels=self.channels, 
            n_channels=self.total_channels)

        self.waveforms = self.waveforms[start_idx:end_idx]
        self.waveforms = np.swapaxes(self.waveforms, 1, 2)

        # cut a window 
        time_length = self.waveforms.shape[2] 
        if n_timesteps is not None:
            start = (time_length - n_timesteps) // 2  # take the middle chunk
            end = start + n_timesteps
            self.waveforms = self.waveforms[:, :, start:end]
            self.n_timesteps = n_timesteps
        else:
            self.n_timesteps = time_length

        self.template_ids = np.unique(self.template_assignments)
        self.n_units = len(self.template_ids)
        self.unit_ids = np.arange(self.n_units, dtype=int)
        self.template_to_unit = {t: u for t, u in zip(self.template_ids, self.unit_ids)}
        self.unit_assignment = np.vectorize(self.template_to_unit.get)(self.template_assignments)
        print('Number of ground truth units:', self.n_units)

        self.waveforms = torch.from_numpy(self.waveforms)
        self.data_shape = self.waveforms.shape
        print('Waveform shape:', self.data_shape)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.unit_assignment[idx] 

        
class SpikeDataGenerator():
    """Generate minibatches of spike waveforms from a spike Dataset object
    """
    def __init__(self, dataset, params=None):
        """
        dataset: a Dataset object of spike waveforms
        """
        self.params = params 
        self.dataset = dataset
        self.data_size = len(dataset)
        self.feature_dim = self.dataset.data_shape[1:]
    
    def generate(self, N, batch_size=1, seed=None):
        """Randomly select a given number of spike waveforms from the dataset
        
        Args:
            N: the number of spikes in each batch
            batch_size: number of minibatches
            seed: random seed
        Returns:
            data: A torch array of shape (batch_size, N, n_channels, n_timesteps) of spike waveforms
            assignments: a numpy array of shape (batch_size, N) of cluster assignements
            spike_time: a numpy array of shape (batch_size, N) of the original spike times
        """
        data = torch.empty([batch_size, N, *self.feature_dim], dtype=torch.float32)
        assignments =  np.empty([batch_size, N], dtype=np.int32)
        template_assignments =  np.empty([batch_size, N], dtype=np.int32)
        dataset_indices =  np.empty([batch_size, N], dtype=np.int32)

        if seed is not None:
            np.random.seed(seed=seed)  
        for i in range(batch_size):
            indices = np.random.choice(self.data_size, N, replace=False)
            data[i], assignments[i] = self.dataset[indices]
            dataset_indices[i] = indices
        
        spike_time = None 
        if hasattr(self.dataset, "spike_time"):
            spike_time = self.dataset.spike_time[dataset_indices]

        return data, assignments, spike_time
