
import numpy as np
import torch
import time
import os, sys
import re, pickle

from ncp import NeuralClustering  # same cluster proportion across batch

from plot_functions import plot_avgs
from utils import relabel, get_parameters
from ncp_sampler import NCP_Sampler

sys.path.append("../spike_sorting")
from spike_data_generator import *
from spike_utils import *
from spike_plot import * 

'''
python main_ncp_spike_fast.py
python main_ncp_spike_fast.py saved_models/partially_trained_checkpoint.pt
'''

model = 'SpikeTemplate'

# loads params from utils.py as a python dictionary
params = get_parameters(model)
params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loads model; parts are specific to spikesoritng; 
ncp_model = NeuralClustering(params).to(params['device'])  

# loading spiking data;
templates = np.load(params['template_file']).transpose(2,1,0)
geom = np.loadtxt(params['geom_file'])
nbr_dist, n_nbrs = params['nbr_dist'], params['n_nbr']

# exclude channels; e.g. if they overlap with test 
if 'channels_exclude' in params:
    chans_exclude = np.load(params['channels_exclude'])
    print("channels to exclude:", chans_exclude)
else:
    chans_exclude = None

# for training find channels with enough neighbours; also excludes explicitly indicated ones
# outputs ~400 channels to be used for training
chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(geom, nbr_dist, n_nbrs, chans_exclude=chans_exclude)

# maps templates to channels; chan_to_template:{chan_no, list of templates on channel}\
# see spike_utils.py file
idx_templates, selected_chans, chan_to_template = select_template_channels(templates, chan_to_nbrs)

# upsample-resample step
templates_upsampled = \
    create_upsampled_templates(templates, idx_templates, selected_chans, upsample=params['temp_upsample'])
print("number of channels used:", len(np.unique(selected_chans[:,0])))

# load a recording for noise simulation 
samplerate = 20000
chunk_len = 60 * samplerate

if params['noise_recordinng_file'].endswith(".bin"):
    noise_recording = load_bin(params['noise_recordinng_file'], 
                    start_time=0, N_CHAN=params['noise_n_chan'], chunk_len=chunk_len, d_type='float32')
elif params['noise_recordinng_file'].endswith(".npy"):
    noise_recording = np.load(params['noise_recordinng_file'])

# geometry of noise recording; we randomly choose a channel to compute noise covariance;
noise_geom = np.loadtxt(params['noise_geom_file'])
_, noise_chan_to_nbrs = get_chan_nbrs(noise_geom, nbr_dist, n_nbrs)
noise_channels = np.stack(list(noise_chan_to_nbrs.values()))
print("noise_recording", noise_recording.shape)

# Yueqi: MFM was used for Ari paper; 
# Note: generate_MFM is not being called; just assigned to cluster-generator
if params['cluster_generator'] == "MFM":
    cluster_generator = generate_MFM
# CRP gives distributions which are something different than approach now used
# where we fix # templates
elif params['cluster_generator'] == "CRP":
    cluster_generator = generate_CRP 
else:
    raise Exception("unknown cluster generator")

print("Using {} cluster generator.".format(params['cluster_generator']))

# spike generator 
data_generator = SpikeGeneratorFromTemplatesCorrNoise(
                            templates_upsampled,  #Note: this array is 4D, last dimension contains multiple shifted version
                            params, 
                            n_timesteps=params['n_timesteps'], ## width of waveform
                            noise_recording=noise_recording,  # raw data for generating noise covariance
                            noise_channels=noise_channels, # same as above
                            noise_thres=3,
                            permute_nbrs=True,  # whether can rotate/flip channels relative centre
                            keep_nbr_order=True,
                            cluster_generator=cluster_generator)

# max number of training templates; poisson process can generate many; 
# current max 12
maxK = params['maxK']

it = -1
# define containers to collect statistics
losses= []       # NLLs    
accs =[]         # Accuracy of the classification prediction
perm_vars = []   # permutation variance
times = []       # Runtime in seconds for each iteration
ratios = []

# nn parameters; 
# adam is a type of optimizer, version of grad descent;
learning_rate = 1e-4
weight_decay = 0.01
optimizer = torch.optim.Adam(ncp_model.parameters() , lr=learning_rate, weight_decay=weight_decay)

#########################
# resume from partially trainded model
if (len(sys.argv) > 1):
    model_checkpoint = sys.argv[1]
    checkpoint = torch.load(model_checkpoint)
    ncp_model.load_state_dict(checkpoint['model_state_dict'])
    ncp_model.to(params['device'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    it = checkpoint['it']

    with open(model_checkpoint[:-3] + "_stats.pkl", 'rb') as f:
        losses, accs, perm_vars = pickle.load(f)
    ratios = [np.sqrt(x)/y for x, y in zip(perm_vars, losses)]

#########################

# no of iterations
it_terminate = 20000

# at these points decrease learning rate by 0.5
milestones = [10000, 17000]
lr_decay = 0.5

# this does the decay; acts as a call-back-function
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_decay, last_epoch=it)

# permutes the samples; e.g. 500 spikes selected are shuffled and re-run
# Yueqi, Cat: this may not be entirely necessary; 
# NOTE: may wish to evaluate the use for this; i.e remove permutations and see if 3 x as many permutations
perms = 3  # Number of permutations for each mini-batch. 
           # In each permutation, the order of the data points is shuffled.         

# for GPU; param that goes into torch
# i.e. takes 32 x 500 spike datasets at a time; 
batch_size = 32 

# mkdir
if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')
if not os.path.isdir('figures'):
    os.mkdir('figures')

model_name = params['model']

# train model
# Note: this object inherits from nn.Module and accordingly has lots of inherited behaviour;
# train() = this is a flag to indicate that params should be initiated for training (diff. behaviour occurs on testing)
ncp_model.train()

# iterate over iterations; 
while True:
    t_start = time.time()
    it += 1
    
    # once every 100 iterations
    if (it % 100 == 0 and it >= 200) or it == it_terminate+1:
        torch.cuda.empty_cache()
        print('\nFraction of GPU Memory used:', torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated(), '\n')            
        
        # plot curve of training accuracy
        plot_avgs(losses, accs, ratios, 50, save_name='./figures/train_avgs_' + model_name + '.pdf')            

        # 
        plot_N = ncp_model.params['Nmax']
        
        # 
        if params['model'] in ['SpikeRaw', 'SpikeTemplate']:
            fname_postfix = "it-{}".format(it)
            
            # perform clustering - test step; 
            # this includes resampling and all the other steps;
            # Note: code below is for training only;
            css, nll, highest_prob, data_arr, gt_labels, reorder, other = \
                cluster_spikes_from_generator(ncp_model, data_generator, NCP_Sampler, N=plot_N, seed=it)
                
            # plot clustering results
            plot_spike_clusters_and_gt_in_rows(
                css, nll, data_arr, gt_labels, topn=2, 
                figdir="./figures", fname_postfix=fname_postfix,
                plot_params={"spacing":1.25, "width":0.9, "vscale":1.5, "subplot_adj":0.9}, 
                downsample=3)
            
            # save some results
            npz_fname = "figures/sample_{}_{}.npz".format(plot_N, fname_postfix)
            np.savez_compressed(npz_fname, css=css, nll=nll, highest_prob=highest_prob,
                    data_arr=data_arr, labels=gt_labels, reorder=reorder, other=other) 
        else:
            raise ValueError("unknown model")
        
    # save checkpoints
    if it % 100 == 0:
        if 'fname' in vars():
            os.remove(fname)
        if 'pickle_fname' in vars():
            os.remove(pickle_fname)
        ncp_model.params['it'] = it
        fname = 'saved_models/'+ model_name + '_' + str(it) + '.pt'        
        pickle_fname = '{}_stats.pkl'.format(fname[:-3])    
        # torch.save(ncp_model, fname)
        torch.save({
            'it': it,
            'model_state_dict': ncp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, fname)

        with open(pickle_fname, 'wb') as f:
            pickle.dump([losses, accs, perm_vars], f)

    if it == it_terminate:
        break

    # data: [batch_size, N, x_dim]
    # cs: shuffled assignments [N], same within the same batch
    # clusters: how many dp in each cluster 
    # K: n_clusters 
    # generate(None, )  # random N for CRP 

    # run data generator every time; 
    if params['model'] in ['SpikeRaw', 'SpikeTemplate']:
        N = np.random.randint(params['Nmin'],params['Nmax'])
        data, cs, _ = data_generator.generate(N, batch_size, maxK=maxK) 
        # print(N)
        # print(data.shape)
        # print(cs.shape)
    else:
        raise ValueError("unknown model")

    loss_values = np.zeros(perms)
    accuracies = np.zeros([N-1,perms])
    
    # The memory requirements change in each iteration according to the random values of N and K.
    # If both N and K are big and out of memory RuntimeError exception might be raised.
    # When this happens, we capture the exception, reduce the batch_size to 3/4 of its value, and try again.
    
    while True:
        try:
            
            loss = 0  # loss over all permutations and N 
            optimizer.zero_grad()    
            
            # 3 permutations indicated above; 
            for perm in range(perms):  # reuse the same data with different shuffles 
                arr = np.arange(N)
                np.random.shuffle(arr)
                cs = cs[arr]
                data = data[:,arr,:]    

                # need to relabel so the clusters can start at 0 after shuffling
                cs = relabel(cs)  # relabel cluster numbers so that they appear in order 
                
                this_loss=0  # loss of this permutation 
                ncp_model.previous_n=0            
                
                # note this runs over each spike - but does full batches at one time
                for n in range(1,N):                
                # n-1 points are already assigned, the point n-th is to be assigned
                    
                    # ncp_model: torch model we defined (most important part of the code)
                    # model takes in data and ground truth labels;
                    # outputs: negative log-likelihood
                    logprobs = ncp_model(data, cs, n)  # [batch_size, K+1]          
                        
                    # cluster labels
                    c = cs[n]  # groud truth cluster for the n'th point 
                    
                    # number of correct cluster assignment / batch size 
                    # converting softwassignment/probability to hard assignement; 
                    accuracies[n-1, perm] = np.sum(np.argmax(logprobs.detach().to('cpu').numpy(),axis=1) == c) / logprobs.shape[0]            
                    
                    # NLL 
                    # note; adding a scallar to a pytorch variable; get back pytorch variable
                    # note: this return an object with methods attached
                    this_loss -= logprobs[:,c].mean()
    
                # average over the N points 
                loss_values[perm] = this_loss.item()/N

                # total loss over all permutations and N 
                loss += this_loss

                # do backward within each permutation 
                # this step computes gradient;
                # note: this variable has a lot of methods attached to it
                this_loss.backward() 
            
            perm_vars.append(loss_values.var())
            losses.append(loss.item()/(N*perms))  # should we also divide by n_perm? 
            # mean acc over permutations and N 
            accs.append(accuracies.mean())

            ratios.append(np.sqrt(perm_vars[-1])/losses[-1])        

            # update all parameters to values from computed gradients
            optimizer.step()
            
            # this tracks milestones
            scheduler.step()  # use scheduler 
    
            lr_curr = optimizer.param_groups[0]['lr']
            print('{0}  N:{1}  Mean NLL:{2:.3f}   Mean Acc:{3:.3f}   Mean Std/NLL: {4:.7f}  Mean Time/Iteration: {5:.1f} lr: {6}'\
                    .format(it, N, np.mean(losses[-50:]), np.mean(accs[-50:]), np.mean(accs[-50:]), (time.time()-t_start), lr_curr))    

            break

        # sometimes synthetic data generate can exceed memory - if something else running in bacgkround 
        # we decrease batch size 
        except RuntimeError:
            bsize = int(.75*data.shape[0])
            if bsize > 2:
                print('RuntimeError handled  ', 'N:', N, 'Trying batch size:', bsize)
                data = data[:bsize,:,:]
            else:
                break





