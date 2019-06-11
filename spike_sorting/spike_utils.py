import numpy as np
import os 

from scipy.signal import resample
from scipy.spatial.distance import pdist, squareform

from collections import defaultdict


def load_bin(fname, start_time, N_CHAN, chunk_len, d_type='float32'):
    # load raw data
    with open(fname, 'rb') as fin:
        if d_type=='float32':
            fin.seek(start_time * 4 * N_CHAN, os.SEEK_SET)
        elif d_type =='int16':
            fin.seek(start_time * 2 * N_CHAN, os.SEEK_SET)
        else:
            print ("uknown data type")
        
        data = np.fromfile(
            fin,
            dtype=d_type,
            count=(chunk_len * N_CHAN)).reshape(chunk_len,N_CHAN)#.astype(np.int32)
    return data


def binary_reader_waveforms(filename, n_channels, n_times, spikes, channels=None, data_type='float32'):
    ''' Reader for loading raw binaries
    
        standardized_filename:  name of file contianing the raw binary
        n_channels:  number of channels in the raw binary recording 
        n_times:  length of waveform 
        spikes: 1D array containing spike times in sample rate of raw data
        channels: load specific channels only
        data_type: float32 for standardized data
    '''

    # ***** LOAD RAW RECORDING *****
    if channels is None:
        wfs = np.zeros((spikes.shape[0], n_times, n_channels), data_type)
        channels = np.arange(n_channels)
    else:
        wfs = np.zeros((spikes.shape[0], n_times, len(channels)), data_type)

    with open(filename, "rb") as fin:
        for ctr,s in enumerate(spikes):
            # index into binary file: time steps * 4  4byte floats * n_channels
            fin.seek(s * 4 * n_channels, os.SEEK_SET)
            wfs[ctr] = np.fromfile(
                fin,
                dtype='float32',
                count=(n_times * n_channels)).reshape(n_times, n_channels)[:,channels]

    fin.close()
    return wfs

def load_waveform(voltage_file, spike_index_all, spike_channel, load_channels=None, n_channels=49, n_times=61):
    """
    voltage_file: standardized voltages file .bin
    spike_index_all: [n_spikes, 2] each row (time, channel)
    spike_channel: which channel on the electrode array 
    """
    # select detected spikes on the desired channel
    spike_index = spike_index_all[spike_index_all[:,1]==spike_channel]

    # spike times in sample time; may need to shift the template a bit
    spikes = spike_index[:,0] - 30

    # read waveforms
    waveforms = binary_reader_waveforms(voltage_file, n_channels, n_times, spikes, load_channels)
    # [n_spikes, n_timesteps, n_channels or len(load_channels)]
    
    return waveforms

def load_waveform_by_spike_time(voltage_file, spike_times, time_offset=-30, load_channels=None, 
                                n_channels=49, n_times=61):
    """
    voltage_file: standardized voltages file .bin
    spike_times: [n_spikes,] 
    time_offset: time offset for spike peak 
    load_channels: which channels on the electrode array to load
    """

    # spike times in sample time; may need to shift the template a bit
    spikes = spike_times + time_offset

    # read waveforms
    waveforms = binary_reader_waveforms(voltage_file, n_channels, n_times, spikes, load_channels)
    # [n_spikes, n_timesteps, n_channels or len(load_channels)]
    
    return waveforms


def vec_to_degree(vec):
    """
    Convert 2d vectors to degrees 
    vec: (dx, dy)
    return:
        counter-clockwise degree from 0 to 360 
    """
    x, y = vec
    if x == 0:
        atan = np.arctan(np.inf * np.sign(y))
    else:
        atan = np.arctan(y/x)
    deg = np.degrees(atan)
    if x < 0:
        deg += 180
    elif y < 0:  # x >= 0, y < 0
        deg += 360
    return deg

    # [60.  0.]: 0.0
    # [30. 60.]: 63.4
    # [ 0. 60.]: 90.0
    # [-30.  60.]: 116.6
    # [-60.   0.]: 180.0
    # [-30. -60.]: 243.4
    # [  0. -60.]: 270.0
    # [ 30. -60.]: 296.6

def find_nbr_channels(geom, channel, n_nbr=None, max_dist=None):
    """
    find neighboring channels according to geom
    n_nbr: number of surrounding channels (including the center)
    """
    dists = np.sqrt((geom[channel,0] - geom[:,0])**2 + (geom[channel,1] - geom[:,1])**2)
    if max_dist is None and n_nbr is None:
        raise Exception("one of max_dist or n_nbr must be provided.")
    
    if max_dist is not None:
        nbr_chan = np.where(dists < max_dist)[0]
        
    if n_nbr is not None:
        nbr_chan = np.argsort(dists)[:n_nbr]  # the order is problematic 
        nbr_chan = np.sort(nbr_chan)
        if max_dist is not None:
            print("using n_nbr rather than max_dist to find neighbors ")
    
    # reorder the center 
    # nbr_chan = np.concatenate([[channel], nbr_chan[nbr_chan != channel]])

    # order the surrounding channels counter-clockwise 
    surround_chans = nbr_chan[nbr_chan != channel]
    vectors = geom[surround_chans] - geom[channel] 
    angles = [vec_to_degree(x) for x in vectors]  # 0 to 360
    sorted_ind = np.argsort(angles)
    nbr_chan = np.concatenate([[channel], surround_chans[sorted_ind]])
    return nbr_chan


def sort_spike_units(data_arr, assignments):
    """
    Args:
        data
        assignments
    Return:
        data_arr: data sorted by assignment
        assignments: sorted assignments
        reorder: np.argsort(assignments)
    """
    reorder = np.argsort(assignments)
    assignments = assignments[reorder]
    data_arr = data_arr[reorder]
    return data_arr, assignments, reorder


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

def get_chan_nbrs(geom, nbr_dist=70, n_nbrs=7, keep_less_nbrs=False, chans_exclude=None):
    """
    Returns:
        chans_with_nbrs: array of channels with n_nbrs in nbr_dist
        chan_to_nbrs: dict(chan: [nbrs])
    """
    channels = list(range(len(geom)))
    if chans_exclude is not None:
        channels = [x for x in channels if not x in chans_exclude]
        print("Excluding {} channels. keep {} channels".format(len(chans_exclude), len(channels)))
    chan_to_nbrs_all = {ch: find_nbr_channels(geom, ch, max_dist=nbr_dist) for ch in channels}

    # select channels that have 7 neighbours;
    if keep_less_nbrs:
        chans_with_nbrs = np.array(sorted([ch for ch, nbrs in chan_to_nbrs_all.items() if len(nbrs)<=n_nbrs]))
    else:
        chans_with_nbrs = np.array(sorted([ch for ch, nbrs in chan_to_nbrs_all.items() if len(nbrs)==n_nbrs]))
    chan_to_nbrs = {x:chan_to_nbrs_all[x] for x in chans_with_nbrs}

    print("Channels with at least {} neighbors (dist <= {}): {}".format(n_nbrs, nbr_dist, len(chan_to_nbrs)))

    return chans_with_nbrs, chan_to_nbrs


def select_template_channels(templates, chans_and_nbrs):
    """
    Args:
        templates: [n_templates, n_channels, n_times]
        channel_and_nbrs: dict(chan: [nbrs]) a subset of channels with its neighbors
    Return:
        idx_templates: array of template ids 
        selected_chans: array of selected channels for each template id 
        chan_to_template: dict{chan: [temp_ids]}
    """
    channels = np.array(list(chans_and_nbrs.keys()))

    # find all units that are on these channels
    max_chans = templates.ptp(2).argmax(1)  # the channels with max peak for each template 
    print ("max chans: ", max_chans.shape)

    idx_templates = np.where(np.isin(max_chans, channels))[0]
    print ("templates in selected chanels {}".format(len(idx_templates)))

    max_chans_selected = max_chans[idx_templates] 
    if len(set([len(chans_and_nbrs[ch]) for ch in max_chans_selected])) == 1:
        selected_chans = np.stack([chans_and_nbrs[ch] for ch in max_chans_selected])
        print ("selectecd chans for all templates: ", selected_chans.shape)
    else:
        selected_chans = [chans_and_nbrs[ch] for ch in max_chans_selected] 
    
    center_channels = [x[0] for x in selected_chans]
    
    chan_to_template = defaultdict(list)
    for ch, temp_id in zip(center_channels, idx_templates):
        chan_to_template[ch].append(temp_id)

    return idx_templates, selected_chans, chan_to_template 

def create_upsampled_templates(templates, idx_templates, selected_chans, upsample=10):
    """
    templates: [n_temp, n_channels, n_times]
    """

    templates_resampled = resample(templates, templates.shape[2] * upsample, axis=2, window=0)
    n_nbrs = selected_chans.shape[1]
    idx=[]
    for k in range(upsample):  # 0,1,2,...,29
        idx.append(np.arange(k, upsample * templates.shape[2], upsample))
    idx = np.array(idx)

    templates_resampled = templates_resampled[:,:,idx]
    
    templates_resampled = templates_resampled[np.tile(idx_templates, (n_nbrs,1)).T, selected_chans]
    templates_resampled = templates_resampled.swapaxes(2,3)
    # [n_temp, n_channels, n_times, n_samples]
    print("templates_downsampled: ", templates_resampled.shape)
    return templates_resampled

def subset_spike_time_by_templates(spike_time_labels, template_ids):
    """
    Args:
        spike_time_labels: [n_spikes, 2] each row (time, template_id)
        template_ids: list of template ids
    Return:
        a subset of spike_time_labels that only contains the target templates.
    """
    return spike_time_labels[np.isin(spike_time_labels[:,1], template_ids)]


def subset_spike_time_by_channel(spike_time_labels, channel_id):
    """
    Args:
        spike_time_labels: [n_spikes, 2] each row (time, channel_id)
        channel_id: int, channel_id
    Return:
        a subset of spike_time_labels from a particular channel.
    """
    return spike_time_labels[spike_time_labels[:,1] == channel_id]


def combine_imgs(img_path_list, save_path, downsample=None):
    images = [Image.open(x) for x in img_path_list]
    widths, heights = zip(*(x.size for x in images))
    total_w = sum(widths)
    total_h = max(heights)

    new_img = Image.new('RGB', (total_w, total_h), color="white")

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    if downsample is not None:
        new_w, new_h = total_w // downsample, total_h // downsample
        new_img.thumbnail((new_w, new_h), Image.ANTIALIAS)
    
    new_img.save(save_path)
    new_img.close()


def combine_imgs_vertical(img_path_list, save_path, downsample=None):
    images = [Image.open(x) for x in img_path_list]
    widths, heights = zip(*(x.size for x in images))
    total_w = max(widths)
    total_h = sum(heights)

    new_img = Image.new('RGB', (total_w, total_h), color="white")

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    if downsample is not None:
        new_w, new_h = total_w // downsample, total_h // downsample
        new_img.thumbnail((new_w, new_h), Image.ANTIALIAS)
    
    new_img.save(save_path)
    new_img.close()
    

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

def template_window(templates, n_timesteps, offset):
    time_length = templates.shape[1]
    start = (time_length - n_timesteps) // 2 + offset  # take the middle chunk with offset
    end = start + n_timesteps
    templates = templates[:, start:end, :]
    return templates


def add_stuff_to_npz(npz_path, new_data):
    assert(isinstance(new_data, dict))
    npz = np.load(npz_path)
    data = {name: npz[name] for name in npz.files}
    for k, v in new_data.items():
        data[k] = v 
    np.savez_compressed(npz_path, **new_data)
