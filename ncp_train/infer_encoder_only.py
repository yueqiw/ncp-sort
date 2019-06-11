 
import numpy as np
import torch
import json
import time
import argparse
import sys, os 
sys.path.append("../spike_sorting")
from ncp_sampler import NCP_Sampler, NCP_SpikeEncoder

from spike_utils import *

from ncp import NeuralClustering
from utils import get_parameters


parser = argparse.ArgumentParser(description='Run NCP encoder on spikes.')
parser.add_argument('input_dir', type=str)

args = parser.parse_args()

# parameters
it_use = 18600
pretrained_path = "./saved_models/SpikeTemplate_{}.pt".format(it_use)

input_dir = args.input_dir
if not os.path.isdir(input_dir):
    raise ValueError("wrong directory")
with open(os.path.join(input_dir, "data_params.json"), "r") as f:
    infer_params = json.load(f)

infer_params['encoder_name'] = 'spike_encoder_it-{}'.format(it_use)

# load model 

model = 'SpikeTemplate'
params = get_parameters(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params['device'] = device

model = NeuralClustering(params)
checkpoint = torch.load(pretrained_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # this is important for spike encoder 

encoder = NCP_SpikeEncoder(model)

# # build encoder only 
# encoder = NCPEncoderSubNetwork(params)
# pretrained_dict = checkpoint['model_state_dict']
# encoder_state = encoder.state_dict()

# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_state}
# # 2. overwrite entries in the existing state dict
# encoder_state.update(pretrained_dict) 
# # 3. load the new state dict
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
    #print(encoded_spikes.shape)

    # save data
    save_fname = os.path.join(data_dir, "{}_encoded_spikes.npz".format(fname))
    np.savez_compressed(save_fname, encoded_spikes=encoded_spikes)
