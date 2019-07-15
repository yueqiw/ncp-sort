
"""Generate real data subsets for clustering 
Usage:
    python -m ncpsort.cluster_real_data.generate_data \
        --output_dir inference_real_data \
        --N 1000 --n_seeds 1 --do_corner_padding

    or  --channel_end_idx 1000 --do_corner_padding
    or  --global_end_idx 90781 (e.g. one minute) --do_corner_padding
"""

import numpy as np
import json
import argparse
import os
from ncpsort.data_generator.real_data import SpikeRawDatasetByChannel, SpikeDataGenerator
from ncpsort.utils.spike_utils import get_chan_nbrs, subset_spike_time_by_channel


parser = argparse.ArgumentParser(
    description='Generate spike data for NCP inference.')
parser.add_argument('--output_dir', type=str, default='inference_real_data',
                    help="name of the directory that stores the generated data.")
parser.add_argument('--N', type=int, default=0,
                    help="how many spikes to subset from the data source. If n=0, extract all spikes")
parser.add_argument('--n_seeds', type=int, default=1,
                    help="if > 1, repeate the data subsetting with different random seeds.")
parser.add_argument('--channel_end_idx', type=int, default=-1,
                    help="if > 0, load the first end_idx spikes from each channel. no random subset.")
parser.add_argument('--global_start_idx', type=int, default=0,
                    help="the spike index of the entire recording to start loading spikes.")
parser.add_argument('--global_end_idx', type=int, default=-1,
                    help="the spike index of the entire recording to stop loading spikes.")
parser.add_argument('--do_corner_padding',
                    action="store_const", const=True, default=False,
                    help="if true, pad zeros for empty neighbor channels (outside the array border).")


if __name__ == "__main__":

    args = parser.parse_args()

    infer_params = {}
    infer_params['data_name'] = args.output_dir
    infer_params["spike_time_raw_no_triage"] = "data/data_49ch/non-triaged_190430/spike_index_all_no-knn-triage.npy"
    infer_params['geom_file'] = 'data/data_49ch/ej49_geometry1.txt'
    infer_params['voltage_file'] = 'data/data_49ch/preprocessing/standardized.bin'
    infer_params['total_n_chan'] = 49
    infer_params['nbr_dist'] = 70
    infer_params['n_nbr'] = 7
    infer_params['n_timesteps'] = 32
    infer_params['time_offset'] = -30
    infer_params['channel_end_idx'] = args.channel_end_idx
    infer_params['global_start_idx'] = args.global_start_idx
    infer_params['global_end_idx'] = args.global_end_idx
    infer_params['do_corner_padding'] = args.do_corner_padding

    if args.N > 0:
        N_default = args.N
        infer_params['data_name'] += '_N-{}'.format(N_default)

    if args.do_corner_padding:
        infer_params['data_name'] += '_pad'

    if args.channel_end_idx != -1:
        N_default = args.channel_end_idx
        infer_params['data_name'] += "_each-ch-first-{}".format(
            args.channel_end_idx)

    if args.global_start_idx != 0 or args.global_end_idx != -1:
        N_default = 'all'
        infer_params['data_name'] += "_global-{}-{}".format(
            args.global_start_idx, args.global_end_idx)

    infer_params['N_default'] = N_default

    # spike time
    spike_time_channel_arr = np.load(infer_params["spike_time_raw_no_triage"])
    spike_time_channel_arr = spike_time_channel_arr[args.global_start_idx:args.global_end_idx]

    # geom and neighbors
    geom = np.loadtxt(infer_params['geom_file'])
    nbr_dist, n_nbrs = infer_params['nbr_dist'], infer_params['n_nbr']
    chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(
        geom, nbr_dist, n_nbrs, keep_less_nbrs=args.do_corner_padding)
    print("{} channels used:".format(len(chans_with_nbrs)))
    print(chans_with_nbrs)
    infer_params['chans_with_nbrs'] = [int(x) for x in chans_with_nbrs]

    output_dir = infer_params['data_name']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    data_dir = os.path.join(output_dir, "data_input")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(os.path.join(output_dir, "data_params.json"), "w") as f:
        json.dump(infer_params, f, indent=2)

    seeds = np.arange(args.n_seeds)

    for ch in chans_with_nbrs:

        spike_time_subset = subset_spike_time_by_channel(
            spike_time_channel_arr, ch)
        nbr_channels = chan_to_nbrs[ch]
        spike_time_subset = spike_time_subset[spike_time_subset[:, 0] +
                                              infer_params['time_offset'] > 0, :]

        ds = SpikeRawDatasetByChannel(
            infer_params['voltage_file'],
            nbr_channels,
            spike_time_subset,
            total_channels=infer_params['total_n_chan'],
            n_timesteps=infer_params['n_timesteps'],
            time_offset=infer_params['time_offset'],
            end_idx=infer_params['channel_end_idx']
        )

        data_generator = SpikeDataGenerator(ds)

        if N_default == 'all' or data_generator.data_size < N_default:
            N = data_generator.data_size
        else:
            N = N_default

        for seed in seeds:
            fname_postfix = "ch-{}_seed-{}_N-{}".format(ch, seed, N)
            print("Generating data on channel {} with seed {}:".format(ch, seed))

            data, gt_labels, spike_time = data_generator.generate(
                N, batch_size=1, seed=seed)
            data_arr = np.array(data[0]).transpose((0, 2, 1))
            if len(nbr_channels) < n_nbrs:
                zero_pad = np.zeros([data_arr.shape[0], data_arr.shape[1],
                                     n_nbrs - data_arr.shape[2]]).astype(np.float32)
                data_arr = np.concatenate([data_arr, zero_pad], axis=2)
            spike_time = spike_time[0]
            gt_labels = gt_labels[0]

            # save data
            npz_fname = os.path.join(data_dir, "{}.npz".format(fname_postfix))
            np.savez_compressed(npz_fname, data_arr=data_arr,
                                spike_time=spike_time, gt_labels=gt_labels)
