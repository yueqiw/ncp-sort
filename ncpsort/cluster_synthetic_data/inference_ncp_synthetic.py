 
"""Perform inference (clustering) using a trained NCP model on synthetic data 
Usage:
    python -m ncpsort.cluster_synthetic_data.inference_ncp_synthetic \
        $input_dir $checkpoint_name --S 150 --beam --topn 2
    or  $input_dir $checkpoint_name --S 1500 --topn 2

    e.g. input_dir = 'inference_synthetic_N-1000', checkpoint_name = 'NCP_18600'
"""

import numpy as np
import torch
import json
import time
import argparse
import os 
from ncpsort.config.train_config import params 
from ncpsort.models.trainer_model import NCP_Trainer
from ncpsort.models.sampler_model import NCP_Sampler
from ncpsort.utils.clustering import cluster_spikes_ncp, get_topn_clusters

parser = argparse.ArgumentParser(description='Run NCP inference on spikes.')
parser.add_argument('input_dir', type=str,
                    help="name of the directory that stores the generated data.") 
parser.add_argument('checkpoint_name', type=str,
                    help="the file name of the trained model checkpoint.")
parser.add_argument('--S', type=int, default=150,
                    help="number of parallel samples (clustering runs).")
parser.add_argument('--beam', action='store_const', default=False, const=True,
                    help="if true, do beam search rather than random sampling.")
parser.add_argument('--topn', type=int, default=2,
                    help="stores the top n clustering results for easy access (actually all results are stored).")


if __name__ == "__main__":
    args = parser.parse_args()

    it_use = args.checkpoint_iter
    pretrained_path = "./saved_models/{}.pt".format(args.checkpoint_name)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError("wrong directory")
    with open(os.path.join(input_dir, "data_params.json"), "r") as f:
        infer_params = json.load(f)

    infer_params['n_parallel_sample'] = n_parallel_sample = args.S
    infer_params['use_beam'] = use_beam =args.beam
    beam_str = "-beam" if use_beam else ""

    infer_params['inference_name'] = 'cluster_S-{}{}_it-{}'.format(n_parallel_sample, beam_str, it_use)

    # load model 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device

    model = NCP_Trainer(params)
    checkpoint = torch.load(pretrained_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # this is important for spike encoder 

    output_dir = os.path.join(input_dir, infer_params['inference_name'])
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    data_dir = os.path.join(output_dir, "data_ncp") 
    if not os.path.isdir(data_dir): os.mkdir(data_dir)

    with open(os.path.join(output_dir, "infer_params.json"), "w") as f:
        json.dump(infer_params, f, indent=2)

    topn = args.topn

    fnames_list = [x.rstrip(".npz") for x in os.listdir(os.path.join(input_dir, "data_input")) if x.endswith(".npz")]
    fnames_list = sorted(fnames_list)

    for fname in fnames_list:

        npz = np.load(os.path.join(input_dir, "data_input", "{}.npz".format(fname)))
        data_arr, gt_labels = npz['data_arr'], npz['gt_labels']

        data_ncp = torch.from_numpy(np.array([data_arr.transpose(0,2,1)]))
        print("Running inference on {}:".format(fname))
        t = time.time()
        clusters, nll, highest_prob = cluster_spikes_ncp(
                model, data_ncp, NCP_Sampler, 
                S=n_parallel_sample, beam=use_beam
            )
        inference_time = time.time() - t 
        print("  time {:4f}".format(inference_time))

        topn_clusters, topn_nll = get_topn_clusters(clusters, nll, topn)

        # save data 
        npz_fname = os.path.join(data_dir, "{}_ncp.npz".format(fname))
        np.savez_compressed(
            npz_fname, clusters=clusters, nll=nll, 
            topn_clusters=topn_clusters, topn_nll=topn_nll, 
            data_arr=data_arr, gt_labels=gt_labels, 
            inference_time=inference_time
        )

