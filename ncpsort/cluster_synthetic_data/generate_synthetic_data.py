
"""Generate synthetic datasets for clustering
Usage:
    python -m ncpsort.cluster_synthetic_data.generate_synthetic_data \
        --output_dir inference_synthetic \
        --N 1000 --n_seeds 10
"""

import numpy as np
import json
import argparse
import os
from ncpsort.data_generator.synthetic_data import SpikeGeneratorFromTemplatesCorrNoise
from ncpsort.data_generator.distributions import CRP_generator, MFM_generator
from ncpsort.utils.spike_utils import get_chan_nbrs, select_template_channels, create_upsampled_templates
from ncpsort.utils.data_readers import load_bin


parser = argparse.ArgumentParser(
    description='Generate spike data for NCP inference.')
parser.add_argument('--output_dir', type=str, default='inference_synthetic',
                    help="name of the directory that stores the generated data.")
parser.add_argument('--N', type=int, default=1000,
                    help="how many spikes to subset from the data source. If n=0, extract all spikes")
parser.add_argument('--n_seeds', type=int, default=10,
                    help="if > 1, repeate the data subsetting with different random seeds.")


if __name__ == "__main__":

    args = parser.parse_args()

    infer_params = {}
    infer_params['N'] = args.N
    infer_params['data_name'] = args.output_dir
    infer_params['template_file'] = "data/data_512ch/templates_post_deconv_pre_merge_2007_512chan.npy"
    infer_params['geom_file'] = 'data/data_512ch/retinal_20min_geometry_2007_512chan.txt'
    infer_params['channels_exclude'] = 'data/data_512ch/chans_512_array_that_map_to_49chan_array.npy'
    infer_params['noise_geom_file'] = 'data/data_49ch/ej49_geometry1.txt'
    infer_params['noise_recordinng_file'] = 'data/data_49ch/preprocessing/standardized.bin'
    infer_params['noise_n_chan'] = 49
    infer_params['nbr_dist'] = 70
    infer_params['n_nbr'] = 7
    infer_params['n_timesteps'] = 32
    infer_params['temp_upsample'] = 5
    infer_params['cluster_generator'] = "MFM"
    infer_params["poisson_lambda"] = 3 - 1
    infer_params["dirichlet_alpha"] = 1
    infer_params["crp_alpha"] = 0.7
    infer_params['maxK'] = 12

    infer_params['data_name'] += '_N-{}'.format(args.N)

    N = args.N

    templates = np.load(infer_params['template_file']).transpose(2, 1, 0)
    geom = np.loadtxt(infer_params['geom_file'])
    nbr_dist, n_nbrs = infer_params['nbr_dist'], infer_params['n_nbr']

    if 'channels_exclude' in infer_params:
        chans_exclude = np.load(infer_params['channels_exclude'])
        print("channels to exclude:", chans_exclude)
    else:
        chans_exclude = None

    chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(
        geom, nbr_dist, n_nbrs, chans_exclude=chans_exclude)

    idx_templates, selected_chans, chan_to_template = select_template_channels(
        templates, chan_to_nbrs)

    templates_upsampled = \
        create_upsampled_templates(
            templates, idx_templates, selected_chans, upsample=infer_params['temp_upsample'])
    # selected_chans: [n_temp, 7]
    print("number of channels used:", len(np.unique(selected_chans[:, 0])))

    samplerate = 20000
    chunk_len = 60 * samplerate
    noise_recording = load_bin(infer_params['noise_recordinng_file'],
                               start_time=0, N_CHAN=infer_params['noise_n_chan'], chunk_len=chunk_len, d_type='float32')
    noise_geom = np.loadtxt(infer_params['noise_geom_file'])
    _, noise_chan_to_nbrs = get_chan_nbrs(noise_geom, nbr_dist, n_nbrs)
    noise_channels = np.stack(list(noise_chan_to_nbrs.values()))
    print("noise_recording", noise_recording.shape)

    if infer_params['cluster_generator'] == "MFM":
        cluster_generator = MFM_generator(
            Nmin=None, Nmax=None, maxK=infer_params['maxK'],
            poisson_lambda=infer_params['poisson_lambda'],
            dirichlet_alpha=infer_params['dirichlet_alpha']
        )
    elif infer_params['cluster_generator'] == "CRP":
        cluster_generator = MFM_generator(
            Nmin=None, Nmax=None, maxK=infer_params['maxK'],
            alpha=infer_params['crp_alpha']
        )
    else:
        raise Exception("unknown cluster generator")

    print("Using {} cluster generator.".format(
        infer_params['cluster_generator']))

    data_generator = SpikeGeneratorFromTemplatesCorrNoise(
        templates_upsampled,
        cluster_generator=cluster_generator,
        noise_recording=noise_recording,
        noise_channels=noise_channels,
        noise_thres=3,
        n_timesteps=infer_params['n_timesteps'],
        permute_nbrs=True,
        keep_nbr_order=True
    )

    maxK = infer_params['maxK']

    output_dir = infer_params['data_name']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    data_dir = os.path.join(output_dir, "data_input")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(os.path.join(output_dir, "data_params.json"), "w") as f:
        json.dump(infer_params, f, indent=2)

    for seed in np.arange(args.n_seeds):
        data, gt_labels, _ = data_generator.generate(
            N, batch_size=1, maxK=maxK, seed=seed)

        fname_postfix = "seed-{}_N-{}".format(seed, N)
        print("Generating data N={} with seed {}:".format(N, seed))

        data_arr = np.array(data[0]).transpose((0, 2, 1))

        # save data
        npz_fname = os.path.join(data_dir, "{}.npz".format(fname_postfix))
        np.savez_compressed(npz_fname, data_arr=data_arr, gt_labels=gt_labels)
