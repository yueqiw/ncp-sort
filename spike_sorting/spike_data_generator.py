import os 
import numpy as np
import torch
from torch.utils.data import Dataset

from spike_utils import * 
from noise_utils import make_noise_torch, noise_cov

def relabel(cs):
    cs = cs.copy()
    d={}
    k=0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k+=1
        cs[i] = d[j]        

    return cs


def generate_MFM(params, N, maxK=None):
    ''' MFM not being used for inference here;
        - just to generate random cluster distributions (i.e. # of spikes per template)
    ''' 
    keep = True
    while keep:
        if N is None or N == 0:  # number of particles, random int Nmin ~ Nmax
            N = np.random.randint(params['Nmin'], params['Nmax'])
        keep = (N == 0)

    # K ~ Pk(k) = Poisson(k-1 | lambda), lambda > 0
    poisson_lambda = params['poisson_lambda'] 
    K_prior = np.random.poisson(poisson_lambda) + 1
    if maxK is not None:
        while K_prior > maxK:
            K_prior = np.random.poisson(poisson_lambda) + 1

    # (pi_1, ..., pi_k) ~ Dirichlet(alpha_1, ..., alpha_k)
    dirichlet_alpha = params['dirichlet_alpha'] 
    cluster_prob = np.random.dirichlet(np.ones(K_prior) * dirichlet_alpha)

    # Z_1, ..., Z_n ~ Cat(pi_1, ..., pi_k)
    cluster_assignments = np.random.choice(np.arange(1, K_prior+1), size=N, p=cluster_prob)
    # if some clusters have low prob, they may not show up in the samples. K_prior != K_sampled
    
    cluster_ids, cluster_count = np.unique(cluster_assignments, return_counts=True)
    K = len(cluster_count)
    clusters = np.zeros(N, dtype=int)
    for i in range(K):
        clusters[i+1] = cluster_count[i]
    return clusters, N, K

def generate_CRP(params,N, no_ones=False, maxK=None):
    
    alpha = params['alpha']   #dispersion parameter of the Chinese Restaurant Process
    keep = True
    
    
    while keep:
        if N is None or N==0:  # number of particles, random int 5-100
            N = np.random.randint(params['Nmin'],params['Nmax'])
            
                
        clusters = np.zeros(N+2)  # 0...N,N+1  how many points in each cluster 
        clusters[0] = 0  # placeholder?
        clusters[1] = 1  # first cluster. we start filling the array here in order to use cumsum below 
        clusters[2] = alpha  # dispersion
        index_new = 2
        for n in range(N-1):     #we loop over N-1 particles because the first particle was assigned already to cluster[1]
            p = clusters/clusters.sum()
            z = np.argmax(np.random.multinomial(1,p))  # random draw from 0...n_clust+1
            if z < index_new:  # existing clusters 
                clusters[z] +=1
            else:  # new cluster 
                clusters[index_new] =1
                index_new +=1
                clusters[index_new] = alpha  # the next new cluster, alpha/(n_samples + alpha)
        
        clusters[index_new] = 0 
        clusters = clusters.astype(np.int32)  # list of int
        
        if no_ones:
            clusters= clusters[clusters!=1]
        N = int(np.sum(clusters))
        K = np.sum(clusters>0)
        keep = N==0 or (maxK is not None and K > maxK)                       
        

    return clusters, N, K


def permute_channels(n_channels, keep_nbr_order=True):
    ch_idx = np.arange(1, n_channels)
    if keep_nbr_order:
        # rotate and flip
        ch_idx = np.roll(ch_idx, np.random.randint(n_channels-1))
        if np.random.randint(2) == 1:
            ch_idx = ch_idx[::-1]
    else:
        # random permute 
        np.random.shuffle(ch_idx)
    ch_idx = np.concatenate([[0], ch_idx])
    return ch_idx 


class SpikeRawDatasetByChannel(Dataset):
    """
    n_nbr channels surroundinng the center.
    """
    def __init__(self, voltage_file, channels, spike_time_labels, total_channels, 
                n_timesteps=None, time_offset=-30, start_idx=0, end_idx=-1, verbose=True):
        """
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
        # [n_spikes, n_nbr, n_timesteps]
        print('Waveform shape:', self.data_shape)
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.unit_assignment[idx] 

class SpikeGeneratorFromTemplatesCorrNoise():
    """
    spatially and temporally correlated noise model
    """
    def __init__(self, templates, params, n_timesteps=32, noise_recording=None, 
                noise_channels=None, noise_thres=3, permute_nbrs=True, keep_nbr_order=True, cluster_generator=generate_CRP):
        """
        templates: [N_temp, n_channels, n_times(, n_samples)]
        """
        self.params = params
        self.templates = templates
        if len(templates.shape) == 3:
            self.n_templates, self.n_channels, time_length = templates.shape 
        elif len(templates.shape) == 4:
            # upsampled-downsampled spikes (last dim)
            self.n_templates, self.n_channels, time_length, self.n_shifts = templates.shape
        
        # cut a window 
        if n_timesteps is not None:
            start = (time_length - n_timesteps) // 2  # take the middle chunk
            end = start + n_timesteps
            self.templates = self.templates[:, :, start:end]
            self.n_timesteps = n_timesteps
        else:
            self.n_timesteps = time_length

        self.templates = torch.from_numpy(self.templates)
        self.permute_nbrs = permute_nbrs
        self.keep_nbr_order = keep_nbr_order
        self.cluster_genenator = cluster_generator
        self.feature_dim = (self.n_channels, self.n_timesteps)
        
        # correlated noise
        # this part computes noise covariance;
        self.kill_window_size = 50
        self.noise_sample_size = 10000
        self.noise_thres = noise_thres
        self.spatial_SIG, self.temporal_SIG = \
                    noise_cov(noise_recording,
                                temporal_size=self.n_timesteps,
                                window_size=self.kill_window_size,  # kill signal within window_size around regions > threshold
                                sample_size=self.noise_sample_size, 
                                threshold=self.noise_thres)
        self.spatial_SIG = torch.from_numpy(self.spatial_SIG.astype(np.float32))
        self.temporal_SIG = torch.from_numpy(self.temporal_SIG.astype(np.float32))
        self.noise_channels = noise_channels


    def generate(self, N=None, batch_size=1, seed=None, maxK=None):
        np.random.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # generates array of labeled events; 
        # 
        clusters, N, K = self.cluster_genenator(self.params, N=N, maxK=maxK)
        # right now the MFM creates the same cluster proportion for all batches 

        # init array to hold spikes for each batch e.g. (32, 500, 32*7) 
        data = torch.empty([batch_size, N, *self.feature_dim], dtype=torch.float32)
        # [batch_size, N, n_channels, n_timesteps]
        
        cumsum = np.cumsum(clusters)
        
        # loop over each batch; 
        # sample a template id from total ground truth temps
        # sample a shift;
        # permute neighbours (rotations or flips only)
        # then save 
        for i in range(batch_size):
            temp_ids = np.random.choice(self.n_templates, size=K, replace = False)  
            for k in range(K):
                tid = temp_ids[k] 
                nk = clusters[k+1]    
                shifts = np.random.choice(self.n_shifts, size=nk, replace=True)  
                if self.permute_nbrs:
                    ch_idx = permute_channels(self.n_channels, self.keep_nbr_order)
                else:
                    ch_idx = np.arange(self.n_channels)
                template_use = self.templates[np.repeat(tid, nk), :, :, shifts]
                data[i, cumsum[k]:cumsum[k+1], :, :] = template_use[:, ch_idx]
        
        # this keeps track of ground truth cluster labels for spikes in each range of spikes 
        cs = np.empty(N, dtype=np.int32)        
        for k in range(K):
            cs[cumsum[k]:cumsum[k+1]]= k

        # data and labels are arranged; need to shuffle; 
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]        
        data = data[:,arr,:,:]
        cs = relabel(cs)

        # add noise
        n_noise_loc = self.noise_channels.shape[0]  # [n_ch, 7]
        for i in range(batch_size):
            # randomly choose a channel from noise rec-array
            noise_ch_idx =  self.noise_channels[np.random.choice(n_noise_loc)]
            # [7]             
            # grab negbhour chans
            surround_ch_idx = np.roll(noise_ch_idx[1:], np.random.randint(self.n_channels-1))
            if np.random.randint(2) == 1:
                surround_ch_idx = surround_ch_idx[::-1]
            noise_ch_idx = np.concatenate([noise_ch_idx[:1], surround_ch_idx])

            # Generate noise for each batch; returns array with unique noise for each spike in batch;
            noise = make_noise_torch(N, self.spatial_SIG[np.ix_(noise_ch_idx, noise_ch_idx)], self.temporal_SIG).transpose(1, 2)
            # [N, n_channels, n_timesteps] 
            data[i] += noise
        
        return data, cs, clusters

        
class SpikeDataGenerator():

    def __init__(self, dataset, params=None):
        """
        dataset: SpikeFeatureDataset or SpikeRawDataset
        """
        self.params = params 
        self.dataset = dataset
        self.data_size = len(dataset)
        self.feature_dim = self.dataset.data_shape[1:]
    
    def generate(self, N, batch_size=1, seed=None):

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

        return data, assignments, spike_time  # other 
