import numpy as np
import os 



def load_bin(fname, start_time, N_CHAN, chunk_len, d_type='float32'):
    """Load Raw data
    """
    with open(fname, 'rb') as fin:
        if d_type=='float32':
            fin.seek(start_time * 4 * N_CHAN, os.SEEK_SET)
        elif d_type =='int16':
            fin.seek(start_time * 2 * N_CHAN, os.SEEK_SET)
        else:
            print ("uknown data type")
        
        data = np.fromfile(
            fin,
            dtype=d_type,
            count=(chunk_len * N_CHAN)).reshape(chunk_len,N_CHAN)#.astype(np.int32)
    return data


def binary_reader_waveforms(filename, n_channels, n_times, spikes, channels=None, data_type='float32'):
    """Reader for loading raw binaries
    
    Args:
        standardized_filename:  name of file contianing the raw binary
        n_channels:  number of channels in the raw binary recording 
        n_times:  length of waveform 
        spikes: 1D array containing spike times in sample rate of raw data
        channels: load specific channels only
        data_type: float32 for standardized data
    """
    if channels is None:
        wfs = np.zeros((spikes.shape[0], n_times, n_channels), data_type)
        channels = np.arange(n_channels)
    else:
        wfs = np.zeros((spikes.shape[0], n_times, len(channels)), data_type)

    with open(filename, "rb") as fin:
        for ctr,s in enumerate(spikes):
            # index into binary file: time steps * 4  4byte floats * n_channels
            fin.seek(s * 4 * n_channels, os.SEEK_SET)
            wfs[ctr] = np.fromfile(
                fin,
                dtype='float32',
                count=(n_times * n_channels)).reshape(n_times, n_channels)[:,channels]

    fin.close()
    return wfs

def load_waveform(voltage_file, spike_index_all, spike_channel, load_channels=None, n_channels=49, n_times=61):
    """Load data

    Args:
        voltage_file: standardized voltages file .bin
        spike_index_all: [n_spikes, 2] each row (time, channel)
        spike_channel: which channel on the electrode array 
    """
    # select detected spikes on the desired channel
    spike_index = spike_index_all[spike_index_all[:,1]==spike_channel]

    # spike times in sample time; may need to shift the template a bit
    spikes = spike_index[:,0] - 30

    # read waveforms
    waveforms = binary_reader_waveforms(voltage_file, n_channels, n_times, spikes, load_channels)
    # [n_spikes, n_timesteps, n_channels or len(load_channels)]
    
    return waveforms

def load_waveform_by_spike_time(voltage_file, spike_times, time_offset=-30, load_channels=None, 
                                n_channels=49, n_times=61):
    """Load data by spike time

    Args:
        voltage_file: standardized voltages file .bin
        spike_times: [n_spikes,] 
        time_offset: time offset for spike peak 
        load_channels: which channels on the electrode array to load
    """
    # spike times in sample time; may need to shift the template a bit
    spikes = spike_times + time_offset

    # read waveforms
    waveforms = binary_reader_waveforms(voltage_file, n_channels, n_times, spikes, load_channels)
    # [n_spikes, n_timesteps, n_channels or len(load_channels)]
    
    return waveforms


def add_data_to_npz(npz_path, new_data):
    assert(isinstance(new_data, dict))
    npz = np.load(npz_path)
    data = {name: npz[name] for name in npz.files}
    for k, v in new_data.items():
        data[k] = v 
    np.savez_compressed(npz_path, **new_data)