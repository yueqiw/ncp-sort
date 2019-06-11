 
import numpy as np
import json
import time
import argparse
import sys, os 
sys.path.append("../spike_sorting")
from spike_data_generator import *
from spike_utils import *
from noise_utils import load_bin


parser = argparse.ArgumentParser(description='Generate spike data for NCP inference.')
parser.add_argument('--N', type=int, default=0)  # n = 0 for all spikes 
parser.add_argument('--n_seeds', type=int, default=1)
args = parser.parse_args()

# --N 1000 --n_seeds 2

model = 'SpikeTemplate'

infer_params = {}
infer_params['template_file'] = "/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/templates_post_deconv_pre_merge_2007_512chan.npy"
infer_params["spike_time_raw_no_triage"] = "/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_190430_non-triaged_and_kilosort/spike_index_all_no-knn-triage.npy"
infer_params['geom_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/retinal_20min_geometry_2007_512chan.txt'
infer_params['channels_exclude'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_512ch/chans_512_array_that_map_to_49chan_array.npy'
infer_params['noise_geom_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_49ch/ej49_geometry1.txt'
infer_params['noise_recordinng_file'] = '/Users/yueqi/Dropbox/lib/discrete_neural_process/data/data_49ch/preprocessing/standardized.bin'
infer_params['noise_n_chan'] = 49
infer_params['nbr_dist'] = 70
infer_params['n_nbr'] = 7
infer_params['n_timesteps'] = 32 
infer_params['temp_upsample'] = 5
infer_params['cluster_generator'] = "MFM"
infer_params["poisson_lambda"] = 3 - 1   
infer_params["dirichlet_alpha"] = 1
infer_params['maxK'] = 12 

templates = np.load(infer_params['template_file']).transpose(2,1,0)
geom = np.loadtxt(infer_params['geom_file'])
nbr_dist, n_nbrs = infer_params['nbr_dist'], infer_params['n_nbr']

if 'channels_exclude' in infer_params:
    chans_exclude = np.load(infer_params['channels_exclude'])
    print("channels to exclude:", chans_exclude)
else:
    chans_exclude = None

chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(geom, nbr_dist, n_nbrs, chans_exclude=chans_exclude)

idx_templates, selected_chans, chan_to_template = select_template_channels(templates, chan_to_nbrs)

templates_upsampled = \
    create_upsampled_templates(templates, idx_templates, selected_chans, upsample=infer_params['temp_upsample'])
# selected_chans: [n_temp, 7]
print("number of channels used:", len(np.unique(selected_chans[:,0])))

samplerate = 20000
chunk_len = 60 * samplerate
noise_recording = load_bin(infer_params['noise_recordinng_file'], 
                    start_time=0, N_CHAN=infer_params['noise_n_chan'], chunk_len=chunk_len, d_type='float32')
noise_geom = np.loadtxt(infer_params['noise_geom_file'])
_, noise_chan_to_nbrs = get_chan_nbrs(noise_geom, nbr_dist, n_nbrs)
noise_channels = np.stack(list(noise_chan_to_nbrs.values()))
print("noise_recording", noise_recording.shape)


if infer_params['cluster_generator'] == "MFM":
    cluster_generator = generate_MFM
elif infer_params['cluster_generator'] == "CRP":
    cluster_generator = generate_CRP 
else:
    raise Exception("unknown cluster generator")

print("Using {} cluster generator.".format(infer_params['cluster_generator']))

data_generator = SpikeGeneratorFromTemplatesCorrNoise(
                            templates_upsampled, 
                            infer_params, 
                            n_timesteps=infer_params['n_timesteps'],
                            noise_recording=noise_recording, 
                            noise_channels=noise_channels,
                            noise_thres=3,
                            permute_nbrs=True,
                            keep_nbr_order=True,
                            cluster_generator=cluster_generator)
maxK = infer_params['maxK']

seeds = np.arange(args.n_seeds)
N = args.N 

infer_params = {}
infer_params['data_name'] = 'inference_49ch_synthetic'
infer_params['N'] = N
# parameters
if args.N > 0:
    N_default = args.N
    infer_params['data_name'] += '_N-{}'.format(N_default)

output_dir = infer_params['data_name']
if not os.path.isdir(output_dir): os.mkdir(output_dir)
data_dir = os.path.join(output_dir, "data_input")
if not os.path.isdir(data_dir): os.mkdir(data_dir)

with open(os.path.join(output_dir, "data_params.json"), "w") as f:
    json.dump(infer_params, f, indent=2)

for seed in seeds:
    data, gt_labels, _ = data_generator.generate(N, batch_size=1, maxK=maxK, seed=seed) 

    fname_postfix = "seed-{}_N-{}".format(seed, N)
    print("Generating data N={} with seed {}:".format(N, seed))
    
    data_arr = np.array(data[0]).transpose((0,2,1))

    # save data
    npz_fname = os.path.join(data_dir, "{}.npz".format(fname_postfix))
    np.savez_compressed(npz_fname, data_arr=data_arr, gt_labels=gt_labels)


