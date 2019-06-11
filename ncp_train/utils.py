

import numpy as np

def get_parameters(model):

    params = {}
        
    if model == 'MNIST':
        params['model'] = 'MNIST'
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        
    elif model == 'Gauss2D':         
        params = {}
        params['model'] = 'Gauss2D'  
        params['alpha'] = .7       # dispersion parameter of the Chinese Restaurant Process
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['sigma_low'] = 0.2
        params['sigma_high'] = 2.5
        params['lambda'] = 10      # mu for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 500
        params['x_dim'] = 5
        params['h_dim'] = 256  # representation for each dp and cluster
        params['g_dim'] = 512  # global representation 
        params['H_dim'] = 128  # intermediate layers 
    
    elif model == 'SpikeRaw':
        params['model'] = 'SpikeRaw'  
        params['n_timesteps'] = 32
        params['n_nbr'] = 7
        params['channel_list'] = list(range(27)) # list(range(27))
        params['resnet_blocks'] = [1,1,1,1]
        params['resnet_planes'] = 32

        params['Nmin'] = 5
        params['Nmax'] = 100 #500
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128

        params['data_dir'] = "/media/yueqi/Data1/data_ml/spike_sorting/ari_clustering/metadata/"
        params['spike_index_file'] = '/media/yueqi/Data1/data_ml/spike_sorting/ari_clustering/preprocessing/spike_index_all.npy'
        params['voltage_file'] = '/media/yueqi/Data1/data_ml/spike_sorting/ari_clustering/preprocessing/standardized.bin'
        params['geom_file'] = '/media/yueqi/Data1/data_ml/spike_sorting/ari_clustering/preprocessing/ej49_geometry1.txt'

    elif model == 'SpikeTemplate':
        params['model'] = 'SpikeTemplate'  
        params['cluster_generator'] = "MFM"  # or CRP
        params['maxK'] = 12  # max number of clusters to generate 

        # CRP
        params['alpha'] = .7       # dispersion parameter of the Chinese Restaurant Process

        # MFM
        params["poisson_lambda"] = 3 - 1   # K ~ Pk(k) = Poisson(lambda) + 1
        params["dirichlet_alpha"] = 1

        params['n_timesteps'] = 32
        params['nbr_dist'] = 70   # nbr distance are 60 or 67 
        params['n_nbr'] = 7

        # ResNet encoder: spike_encoder.py params;
        params['resnet_blocks'] = [1,1,1,1]
        params['resnet_planes'] = 32

        # N ~ unif(Nmin, Nmax); total number of data points;
        # at training time up to 500 usually; testing can do up to 2000
        params['Nmin'] = 200
        params['Nmax'] = 500 #500

        # NCP layer dimensions; NN architecture; ok to leave fixed for now;
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128

        # template config; 512chan templates are used; upsample rate up to 5;
        params['temp_n_chan'] = 512
        params['template_data'] = "512ch"
        params['temp_upsample'] = 5

        params['template_file'] = "/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/templates_post_deconv_pre_merge_2007_512chan.npy"
        params['geom_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/retinal_20min_geometry_2007_512chan.txt'
        # channels to exclude as they contain testing data;
        params['channels_exclude'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/chans_512_array_that_map_to_49chan_array.npy'
            
        # noise config parameters; 
        # entire recording
        # params['noise_recordinng_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_49ch/preprocessing/standardized.bin'
        # only the first 60s - saved already
        params['noise_recordinng_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_49ch/standardized_2007_49chan_60sec.npy'
        params['noise_geom_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_49ch/ej49_geometry1.txt'
        params['noise_n_chan'] = 49
    else:
        raise NameError('Unknown model '+ model)
        
    return params


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

        



    



