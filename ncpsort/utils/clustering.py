
import numpy as np 
import torch 

def cluster_spikes_ncp(model, data, sampler, S=1500, beam=False):
    ncp_sampler = sampler(model, data)
    css, nll = ncp_sampler.sample(S, beam=beam)
    highest_prob = np.exp(-nll.min())

    return css, nll, highest_prob

def cluster_spikes_from_generator(model, data_generator, sampler, N=100, seed=None, S=1500, beam=False):
    """
    posterior sampling 
    S=1500, beam=False or S=15, beam=True
    """
    
    # generate synthetic data; 
    data, assignments, other = data_generator.generate(N, batch_size=1, seed=seed)
    data_arr = np.array(data[0]).transpose((0,2,1))
    if len(assignments.shape) > 1:
        assignments = assignments[0]

    # inference 
    ncp_sampler = sampler(model, data)
    css, nll = ncp_sampler.sample(S, beam=beam)
    highest_prob = np.exp(-nll.min())

    return css, nll, highest_prob, data_arr, assignments, other 


def get_best_cluster(css, nll):
    nll_best = nll.min()
    best = css[nll==nll_best]
    if len(best.shape) == 2:
        best = best[0]
    return best, nll_best

def get_topn_clusters(css, nll, topn=1):
    sorted_nll = np.sort(list(set(nll)))
    topn_css = []
    topn_nll = []
    for i in range(topn): 
        snll= sorted_nll[i]
        r = np.nonzero(nll==snll)[0][0]
        cs = css[r,:]
        topn_css.append(cs)
        topn_nll.append(snll)
    topn_css = np.array(topn_css)
    topn_nll = np.array(topn_nll)
    return topn_css, topn_nll


def relabel(cs):
    cs = cs.copy()
    d={}
    k=0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k+=1
        cs[i] = d[j]        

    return cs