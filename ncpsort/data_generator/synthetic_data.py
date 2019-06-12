
import numpy as np
import torch

from ncpsort.utils.clustering import relabel
from ncpsort.data_generator.noise import make_noise_torch, noise_cov


class SpikeGeneratorFromTemplatesCorrNoise():
    """
    spatially and temporally correlated noise model
    """
    def __init__(self, templates, cluster_generator, n_timesteps=32, noise_recording=None, 
                noise_channels=None, noise_thres=3, permute_nbrs=True, keep_nbr_order=True):
        """
        templates: [N_temp, n_channels, n_times(, n_samples)]
        """

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
        clusters, N, K = self.cluster_genenator.generate(N)
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