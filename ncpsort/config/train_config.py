
params = {
    'model': 'SpikeTemplate',  # model name
    'cluster_generator': "MFM",  # or CRP
    'maxK': 12,  # max number of clusters to generate 

    # MFM
    "poisson_lambda": 3 - 1,  # K ~ Pk(k) = Poisson(lambda) + 1
    "dirichlet_alpha": 1,  # prior for cluster proportions

    # CRP
    'crp_alpha': .7,  # dispersion parameter of CRP 

    # data shape 
    'n_timesteps': 32,  # width of each spike
    'n_channels': 7,  # number of local channels/units

    # ResNet encoder: parameters for spike_encoder.py 
    'resnet_blocks': [1,1,1,1],
    'resnet_planes': 32,

    # number of data points for training, N ~ unif(Nmin, Nmax)
    'Nmin': 200,
    'Nmax': 500, 

    # neural net architecture for NCP
    'h_dim': 256,
    'g_dim': 512,
    'H_dim': 128,
}



        



    



