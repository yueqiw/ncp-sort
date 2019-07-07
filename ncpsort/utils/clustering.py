
import numpy as np
import torch


def cluster_spikes_ncp(model, data, sampler, S=1500, beam=False):
    """Probabilistic clustering of spike waveform data with NCP

    Common parameter choices: S=1500, beam=False or S=15, beam=True

    Args:
        model: a trained model from trainer_model.NCP_trainer()
        data: a torch array of shape (1, N, n_channels, n_timesteps) of spike waveforms
        sampler: a Sampler class clusters the data using the trained model, e.g. sampler_model.NCP_Sampler
        S: the number of parallel samples
        beam: whether to do beam search 
    Returns:
        clusters: a numpy array of shape (S, N) of S parallel samples of N cluster labels
        nll: a numpy array of shape (S,) of the negative log-likelihood of each sample
        highest_prob: highest probability among the clusters 
    """
    ncp_sampler = sampler(model, data)
    clusters, nll = ncp_sampler.sample(S, beam=beam)
    highest_prob = np.exp(-nll.min())
    return clusters, nll, highest_prob


def cluster_spikes_from_generator(model, data_generator, sampler,
                                  N=100, seed=None, S=1500, beam=False):
    """ProbabProbabilistic clustering of spike waveforms produced from a data generator
    
    Common parameter choices: S=1500, beam=False or S=15, beam=True

    Args:
        model: a trained model from trainer_model.NCP_trainer()
        data_generator: a parameterized data generator (instance of a class)
        sampler: a Sampler class clusters the data using the trained model, e.g. sampler_model.NCP_Sampler
        N: the number of generated data points 
        seed: random seed
        S: the number of parallel samples
        beam: whether to do beam search 
    Returns:
        clusters: a numpy array of shape (S, N) of S parallel samples of N cluster labels
        nll: a numpy array of shape (S,) of the negative log-likelihood of each sample
        highest_prob: highest probability among the clusters 
        data_arr: data produced from the data generator 
        assignments: cluster assignments 
        extra_slot: the extra info produced from the data generator (e.g. spike times, etc)
    """

    # generate synthetic data;
    data, assignments, extra_slot = \
        data_generator.generate(N, batch_size=1, seed=seed)
    data_arr = np.array(data[0]).transpose((0, 2, 1))
    if len(assignments.shape) > 1:
        assignments = assignments[0]

    # inference
    ncp_sampler = sampler(model, data)
    clusters, nll = ncp_sampler.sample(S, beam=beam)
    highest_prob = np.exp(-nll.min())

    return clusters, nll, highest_prob, data_arr, assignments, extra_slot


def get_best_cluster(clusters, nll):
    """Get the cluster assignment with the highest probability 
    """
    nll_best = nll.min()
    best = clusters[nll == nll_best]
    if len(best.shape) == 2:
        best = best[0]
    return best, nll_best


def get_topn_clusters(clusters, nll, topn=1):
    """Get the top-n cluster assignments sorted by probability 
    """
    sorted_nll = np.sort(list(set(nll)))
    topn_clusters = []
    topn_nll = []
    for i in range(topn):
        snll = sorted_nll[i]
        r = np.nonzero(nll == snll)[0][0]
        cs = clusters[r, :]
        topn_clusters.append(cs)
        topn_nll.append(snll)
    topn_clusters = np.array(topn_clusters)
    topn_nll = np.array(topn_nll)
    return topn_clusters, topn_nll


def relabel(cs):
    """Relabel the clusters 
    """
    cs = cs.copy()
    d = {}
    k = 0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k += 1
        cs[i] = d[j]

    return cs
