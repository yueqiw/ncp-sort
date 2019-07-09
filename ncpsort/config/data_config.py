
data_config = {
    'nbr_dist': 70,   # nbr distance are 60 or 67 

    # template config
    'temp_n_chan': 512,
    'template_data': "512ch",
    'temp_upsample': 5,

    # training data
    'template_file': "data/data_512ch/templates_post_deconv_pre_merge_2007_512chan.npy",
    'geom_file': 'data/data_512ch/retinal_20min_geometry_2007_512chan.txt',
    # channels to exclude from training
    'channels_exclude': 'data/data_512ch/chans_512_array_that_map_to_49chan_array.npy',
        
    # only the first 60s 
    'noise_recordinng_file': 'data/data_49ch/standardized_2007_49chan_60sec.npy',
    'noise_geom_file': 'data/data_49ch/ej49_geometry1.txt',
}