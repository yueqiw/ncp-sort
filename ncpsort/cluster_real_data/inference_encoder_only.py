 
"""Run the spike encoder of a trained NCP model
Usage:
    python -m ncpsort.cluster_real_data.inference_encoder_only \
        input_dir checkpoint_iter 

    e.g. input_dir = inference_real_data_N-1000_pad, checkpoint_iter = 18600
"""

import numpy as np
import torch
import json
import time
import argparse
import os 
from ncpsort.config.train_config import params
from ncpsort.models.trainer_model import NCP_Trainer
from ncpsort.models.spike_encoder import NCP_SpikeEncoder

parser = argparse.ArgumentParser(description='Run NCP encoder on spikes.')
parser.add_argument('input_dir', type=str,
                    help="name of the directory that stores the generated data.")
parser.add_argument('checkpoint_iter', type=int,
                    help="the final iteration of the trained model checkpoint.")

if __name__ == "__main__":
    args = parser.parse_args()

    it_use = args.checkpoint_iter
    pretrained_path = "./saved_models/{}_{}.pt".format(params['model'], it_use)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError("wrong directory")
    with open(os.path.join(input_dir, "data_params.json"), "r") as f:
        infer_params = json.load(f)

    infer_params['encoder_name'] = 'spike_encoder_it-{}'.format(it_use)

    # load model 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['device'] = device

    model = NCP_Trainer(params)
    checkpoint = torch.load(pretrained_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # this is important for spike encoder 

    encoder = NCP_SpikeEncoder(model)

    # # alternative way to build encoder only 
    # encoder = NCPEncoderSubNetwork(params)
    # pretrained_dict = checkpoint['model_state_dict']
    # encoder_state = encoder.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_state}
    # # overwrite entries in the existing state dict
    # encoder_state.update(pretrained_dict) 
    # # load the new state dict
    # encoder.load_state_dict(pretrained_dict)
    # encoder.to(device)
    # encoder.eval()  # this is important for spike encoder 
    # print(encoder)

    output_dir = os.path.join(input_dir, infer_params['encoder_name'])
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    data_dir = os.path.join(output_dir, "data_encoder") 
    if not os.path.isdir(data_dir): os.mkdir(data_dir)

    with open(os.path.join(output_dir, "encoder_params.json"), "w") as f:
        json.dump(infer_params, f, indent=2)

    fnames_list = [x.rstrip(".npz") for x in os.listdir(os.path.join(input_dir, "data_input")) if x.endswith(".npz")]
    fnames_list = sorted(fnames_list)

    for fname in fnames_list:
        npz = np.load(os.path.join(input_dir, "data_input", "{}.npz".format(fname)))
        data_arr = npz['data_arr']

        data_ncp = torch.from_numpy(np.array([data_arr.transpose(0,2,1)]))
        print("Running encoder on {}:".format(fname))

        encoded_spikes = encoder.encode(data_ncp)
        encoded_spikes = encoded_spikes.cpu().detach().numpy()
        encoded_spikes = encoded_spikes[0]

        # save data
        save_fname = os.path.join(data_dir, "{}_encoded_spikes.npz".format(fname))
        np.savez_compressed(save_fname, encoded_spikes=encoded_spikes)

