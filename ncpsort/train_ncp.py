
'''Train NCP model on synthetic data 
Usage:
    python -m ncpsort.train_ncp --n_iter 10000
    python -m ncpsort.train_ncp --n_iter 10000 --saved_checkpoint saved_models/NCP_5000.pt
'''

import numpy as np
import torch
import time
import os
import pickle
import argparse

from ncpsort.models.trainer_model import NCP_Trainer
from ncpsort.models.sampler_model import NCP_Sampler

from ncpsort.data_generator.synthetic_data import SpikeGeneratorFromTemplatesCorrNoise
from ncpsort.data_generator.distributions import MFM_generator, CRP_generator

from ncpsort.utils.spike_utils import get_chan_nbrs, select_template_channels, create_upsampled_templates
from ncpsort.utils.clustering import cluster_spikes_from_generator, relabel
from ncpsort.utils.plotting import plot_avgs, plot_spike_clusters_and_gt_in_rows

from ncpsort.config.train_config import params
from ncpsort.config.data_config import data_config

parser = argparse.ArgumentParser(description="Train NCP model on synthetic data.")
parser.add_argument('--n_iter', type=int, default=10000,
                    help="number of training iterations.")
parser.add_argument('--saved_checkpoint', type=str, default=None,
                    help="if provided (as file path), continue training from the checkpoint.")

def get_data_generator(params, data_config):

    # loading spiking data
    templates = np.load(data_config['template_file']).transpose(2, 1, 0)
    geom = np.loadtxt(data_config['geom_file'])
    nbr_dist, n_nbrs = data_config['nbr_dist'], params['n_channels']

    # exclude channels (e.g. for testing)
    if 'channels_exclude' in params:
        chans_exclude = np.load(data_config['channels_exclude'])
        print("channels to exclude:", chans_exclude)
    else:
        chans_exclude = None

    # find channels with neighbours
    chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(
        geom, nbr_dist, n_nbrs, chans_exclude=chans_exclude)

    # maps templates to channels 
    idx_templates, selected_chans, chan_to_template = select_template_channels(
        templates, chan_to_nbrs)
    # chan_to_template: chan_id => templates on that channel

    # upsample/resample the templates at higher frequency (shift and downsample later)
    templates_upsampled = \
        create_upsampled_templates(
            templates, idx_templates, selected_chans, upsample=data_config['temp_upsample'])
    # shape (n_templates, n_channels, n_timesteps, n_shifts). multiple shifts in the last dimension

    print("number of channels used:", len(np.unique(selected_chans[:, 0])))

    # load a recording for noise simulation
    # samplerate = 20000
    # chunk_len = 60 * samplerate
    # noise_n_chan = 49
    # noise_recording = load_bin(params['noise_recordinng_file'],
    #                 start_time=0, N_CHAN=noise_n_chan, chunk_len=chunk_len, d_type='float32')

    noise_recording = np.load(data_config['noise_recordinng_file'])

    # geometry of noise recording used to compute noise covariance;
    noise_geom = np.loadtxt(data_config['noise_geom_file'])
    _, noise_chan_to_nbrs = get_chan_nbrs(noise_geom, nbr_dist, n_nbrs)
    noise_channels = np.stack(list(noise_chan_to_nbrs.values()))
    print("noise_recording", noise_recording.shape)

    # in CRP, the number of clusters vary with different N
    # in MFM, the number of clusters does not depend on N, which is more consistent with spike data
    # thus, MFM is currently a better generative model than CRP 
    if params['cluster_generator'] == "MFM":
        cluster_generator = MFM_generator(
            Nmin=params['Nmin'], Nmax=params['Nmax'], maxK=params['maxK'],
            poisson_lambda=params['poisson_lambda'], dirichlet_alpha=params['dirichlet_alpha']
        )
    elif params['cluster_generator'] == "CRP":
        cluster_generator = CRP_generator(
            Nmin=params['Nmin'], Nmax=params['Nmax'], maxK=params['maxK'], alpha=params['crp_alpha']
        )
    else:
        raise Exception("unknown cluster generator")

    print("Using {} cluster generator.".format(params['cluster_generator']))

    # spike generator
    data_generator = SpikeGeneratorFromTemplatesCorrNoise(
        templates_upsampled,
        cluster_generator=cluster_generator,
        n_timesteps=params['n_timesteps'],  # width of waveform
        noise_recording=noise_recording,  # raw data for generating noise covariance
        noise_channels=noise_channels,  # same as above
        noise_thres=3,
        permute_nbrs=True,  # whether can rotate/flip neighbor channels 
        keep_nbr_order=True
    )
    return data_generator


if __name__ == "__main__":

    args = parser.parse_args()

    # loads params from train_config.py as a python dictionary
    params['device'] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    data_generator = get_data_generator(params, data_config)

    # parameterize the model for training 
    ncp_model = NCP_Trainer(params).to(params['device'])

    # max number of training clusters (clip the long tail of Poisson)
    maxK = params['maxK']

    it = -1

    # define containers to collect statistics
    losses = []      # NLLs
    accs = []        # Accuracy of the classification prediction
    perm_vars = []   # permutation variance
    times = []       # Runtime in seconds for each iteration
    ratios = []

    # training parameters;
    learning_rate = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.Adam(
        ncp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #########################
    # resume from partially trainded model
    if args.saved_checkpoint is not None:
        model_checkpoint = args.saved_checkpoint
        checkpoint = torch.load(model_checkpoint)
        ncp_model.load_state_dict(checkpoint['model_state_dict'])
        ncp_model.to(params['device'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it = checkpoint['it']

        with open(model_checkpoint[:-3] + "_stats.pkl", 'rb') as f:
            losses, accs, perm_vars = pickle.load(f)
        ratios = [np.sqrt(x)/y for x, y in zip(perm_vars, losses)]
    #########################

    # total number of iterations
    it_terminate = args.n_iter

    # at these points decrease learning rate by lr_decay
    milestones = [it_terminate // 2, it_terminate * 7 // 8]
    lr_decay = 0.5

    # a callback-function to schedule the LR decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=lr_decay, last_epoch=it)

    # permutes the samples in each mini-batch
    perms = 3  
    # In each permutation, the order of spikes is shuffled and the forward/backward pass is re-run.
    # this may not be entirely necessary for training but help keep track of NLL variance;
    # NOTE: may wish to evaluate the use for this

    batch_size = 32

    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
    if not os.path.isdir('train_log'):
        os.mkdir('train_log')

    model_name = params['model_name']

    # set the model parameters to the training mode (vs. eval mode)
    ncp_model.train()

    # training iterations
    while True:
        t_start = time.time()
        it += 1

        # plot training curve and clustering results to monitor training progress 
        # skip the first 200 iterations (bad models are slow at inference time)
        if (it % 100 == 0 and it >= 200) or it == it_terminate+1:
            torch.cuda.empty_cache()

            # plot curve of training accuracy
            plot_avgs(losses, accs, ratios, 50,
                      save_name='./figures/train_avgs_' + model_name + '.pdf')

            plot_N = ncp_model.params['Nmax']
            fname_postfix = "it-{}".format(it)

            # perform clustering -- inference step;
            css, nll, highest_prob, data_arr, gt_labels, reorder, _ = \
                cluster_spikes_from_generator(
                    ncp_model, data_generator, NCP_Sampler, N=plot_N, seed=it)

            # plot clustering results
            plot_spike_clusters_and_gt_in_rows(
                css, nll, data_arr, gt_labels, topn=2,
                figdir="./figures", fname_postfix=fname_postfix,
                plot_params={"spacing": 1.25, "width": 0.9,
                                "vscale": 1.5, "subplot_adj": 0.9},
                downsample=3)

            # save results
            npz_fname = "figures/sample_{}_{}.npz".format(
                plot_N, fname_postfix)
            np.savez_compressed(npz_fname, css=css, nll=nll, highest_prob=highest_prob,
                                data_arr=data_arr, labels=gt_labels, reorder=reorder, other=other)

        # save model checkpoints
        if it % 100 == 0:
            # remove previous checkpoints 
            if 'fname' in vars():
                os.remove(fname)
            if 'pickle_fname' in vars():
                os.remove(pickle_fname)
            ncp_model.params['it'] = it
            fname = 'saved_models/' + model_name + '_' + str(it) + '.pt'
            pickle_fname = '{}_stats.pkl'.format(fname[:-3])
            torch.save({
                'it': it,
                'model_state_dict': ncp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, fname)

            with open(pickle_fname, 'wb') as f:
                pickle.dump([losses, accs, perm_vars], f)

        # terminate training 
        if it == it_terminate:
            break

        # generate a new batch of data 
        N = np.random.randint(params['Nmin'], params['Nmax'])
        data, cs, _ = data_generator.generate(N, batch_size, maxK=maxK)
        # data: [batch_size, N, n_channels, n_timesteps]
        # cs: shuffled assignments [N], same within the same batch

        loss_values = np.zeros(perms)
        accuracies = np.zeros([N-1, perms])

        # The memory usage changes in each iteration according to the random values of N and K.
        # If both N and K are big ,out of memory RuntimeError exception might be raised.
        # When this happens, we capture the exception, reduce the batch_size and rerun 

        while True:
            try:
                loss = 0  # loss over all permutations and N
                optimizer.zero_grad()

                # iterate through permutations
                for perm in range(perms):  # reuse the same data with different permutations
                    arr = np.arange(N)
                    np.random.shuffle(arr)

                    cs = cs[arr]
                    data = data[:, arr, :]

                    # relabel so the clusters numbers appear in order (start from 0) after shuffling
                    cs = relabel(cs)

                    this_loss = 0  # loss of this permutation
                    ncp_model.previous_n = 0

                    # iterate through one spike at a time (the entire batch runs in parallel)
                    for n in range(1, N):
                        # Elements 1,2,...,n-1 are already assigned 
                        # The n-th element is to be assigned. 
                        # Elements after the n-th are not assigned

                        # the model takes in data and ground truth labels
                        # runs forward pass and outputs log-prob for assigning to each cluster
                        logprobs = ncp_model(data, cs, n)  # [batch_size, K+1]

                        c = cs[n]  # groud truth cluster labels for the n-th point

                        # number of correct cluster assignment / batch size
                        # convert soft assignment/probability to hard assignement;
                        accuracies[n-1, perm] = np.sum(np.argmax(logprobs.detach().to(
                            'cpu').numpy(), axis=1) == c) / logprobs.shape[0]

                        # NLL
                        this_loss -= logprobs[:, c].mean()

                    # average over the N points
                    loss_values[perm] = this_loss.item()/N

                    # total loss over all permutations and N
                    loss += this_loss

                    # compute gradient within each permutation (without updating parameters)
                    this_loss.backward()

                perm_vars.append(loss_values.var())
                losses.append(loss.item()/(N*perms))
                # mean acc over permutations and N
                accs.append(accuracies.mean())

                ratios.append(np.sqrt(perm_vars[-1])/losses[-1])

                # update all parameters by computed gradients
                optimizer.step()

                # callback for updating learning rate at milestones
                scheduler.step()  

                lr_curr = optimizer.param_groups[0]['lr']
                print('{0}  N:{1}  Mean NLL:{2:.3f}   Mean Acc:{3:.3f}   Mean Std/NLL: {4:.7f}  Mean Time/Iteration: {5:.1f} lr: {6}'
                      .format(it, N, np.mean(losses[-50:]), np.mean(accs[-50:]), np.mean(accs[-50:]), (time.time()-t_start), lr_curr))
                # if no memory error
                break

            # In case of memory overflow (e.g. some other programs running)
            # we decrease batch size without interrupting the training 
            except RuntimeError:
                bsize = int(.75*data.shape[0])
                if bsize > 2:
                    print('RuntimeError handled  ', 'N:',
                          N, 'Trying batch size:', bsize)
                    data = data[:bsize, :, :]
                else:
                    break



