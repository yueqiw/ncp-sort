
params = {
    'model': 'SpikeTemplate',
    'cluster_generator': "MFM",  # or CRP
    'maxK': 12,  # max number of clusters to generate 

    # CRP
    'crp_alpha': .7,       # dispersion parameter of the Chinese Restaurant Process

    # MFM
    "poisson_lambda": 3 - 1,   # K ~ Pk(k) = Poisson(lambda) + 1
    "dirichlet_alpha": 1,

    'n_timesteps': 32,
    'n_channels': 7,

    # ResNet encoder: spike_encoder.py params;
    'resnet_blocks': [1,1,1,1],
    'resnet_planes': 32,

    # N ~ unif(Nmin, Nmax); total number of data points;
    # at training time up to 500 usually; testing can do up to 2000
    'Nmin': 200,
    'Nmax': 500, #500

    # NCP layer dimensions; NN architecture; ok to leave fixed for now;
    'h_dim': 256,
    'g_dim': 512,
    'H_dim': 128,

}



        



    



