import numpy as np
import os

from scipy.signal import resample
from scipy.spatial.distance import pdist, squareform

from collections import defaultdict


def get_chan_nbrs(geom, nbr_dist=70, n_nbrs=7, keep_less_nbrs=False, chans_exclude=None):
    """Get the channels with n_nbrs neightbors within nbr_dist radius on the electrode array. 

    Returns an array of such channels and a mapping of each channel to its neighbors. 
    Have the option to keep chennels with less than n_nbrs neightbors. 

    Args:
        geom: A 2-D numpy array of shape (n_channels, 2) representing the XY coordinates of all channels.
        nbr_dist: The maximum distance of surrounding channels (if n_nbr is None).
        n_nbrs: The number of surrounding channels (including the center). Ignores max_dist.
        keep_less_nbrs: a boolean choice of whether to keep channels with < n_nbr neighbors. 
        chans_exclude: an array of channels IDs to exclude
    Returns:
        chans_with_nbrs: array of channels with n_nbrs (or less) neighbors within nbr_dist radius. 
        chan_to_nbrs: a dict mapping each channel to a list of neighbor channels. 
    """
    channels = list(range(len(geom)))
    if chans_exclude is not None:
        channels = [x for x in channels if not x in chans_exclude]
        print("Excluding {} channels. keep {} channels".format(
            len(chans_exclude), len(channels)))

    chan_to_nbrs_all = {ch: find_nbr_channels(
        geom, ch, max_dist=nbr_dist) for ch in channels}

    # select channels that have n_nbrs (or less) neighbours;
    if keep_less_nbrs:
        chans_with_nbrs = np.array(
            sorted([ch for ch, nbrs in chan_to_nbrs_all.items() if len(nbrs) <= n_nbrs]))
    else:
        chans_with_nbrs = np.array(
            sorted([ch for ch, nbrs in chan_to_nbrs_all.items() if len(nbrs) == n_nbrs]))
    chan_to_nbrs = {x: chan_to_nbrs_all[x] for x in chans_with_nbrs}

    print("Channels with at least {} neighbors (dist <= {}): {}".format(
        n_nbrs, nbr_dist, len(chan_to_nbrs)))

    return chans_with_nbrs, chan_to_nbrs


def find_nbr_channels(geom, channel, n_nbr=None, max_dist=None):
    """Find neighbors of a given channel according to the geometry of electrode array. 

    Includes the center channel. Also reorders the surrounding channels counter-clockwise. 

    Args:
        geom: A 2-D numpy array of shape (n_channels, 2) representing the XY coordinates of all channels.
        channel: The id of the center channel.
        n_nbr: The number of surrounding channels (including the center). Ignores max_dist.
        max_dist: The maximum distance of surrounding channels (if n_nbr is None).

    Returns:
        A 1-D numpy array of surrouding channels, with the center channel itself as the 1st element. 
    """
    dists = np.sqrt((geom[channel, 0] - geom[:, 0]) **
                    2 + (geom[channel, 1] - geom[:, 1])**2)
    if max_dist is None and n_nbr is None:
        raise Exception("one of max_dist or n_nbr must be provided.")

    if max_dist is not None:
        nbr_chan = np.where(dists < max_dist)[0]

    if n_nbr is not None:
        nbr_chan = np.argsort(dists)[:n_nbr]  
        nbr_chan = np.sort(nbr_chan)
        if max_dist is not None:
            print("using n_nbr rather than max_dist to find neighbors ")

    # order the surrounding channels counter-clockwise
    surround_chans = nbr_chan[nbr_chan != channel]
    vectors = geom[surround_chans] - geom[channel]
    angles = [vec_to_degree(x) for x in vectors]   # 0 to 360 degrees 
    sorted_ind = np.argsort(angles)
    nbr_chan = np.concatenate([[channel], surround_chans[sorted_ind]])
    return nbr_chan


def sort_spike_units(data_arr, assignments):
    """Sort the spike waveform array according to cluster assignments. 

    Args:
        data_arr: A 2-D numpy array of shape (n_samples, waveform_len)
        assignments: A 1-D numpy array (length = n_samples) of numeric cluster assignments   
    Returns:
        data_arr: data_arr with the first dimension sorted by assignments
        assignments: sorted assignments
        reorder: reordering index of the sorted assignments
    """
    reorder = np.argsort(assignments)
    assignments = assignments[reorder]
    data_arr = data_arr[reorder]
    return data_arr, assignments, reorder


def vec_to_degree(vec):
    """Convert 2D vectors to angle in degrees. 
    
    Args:
        vec: A 2-D numpy array of shape (n_pts, 2). Each row represents the relative 
            distance (dx, dy) of each point from the center.
    Returns:
        A 1-D numpy array of length n_pts representing the counter-clockwise 
            degree from 0 to 360 for each point.
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


def select_template_channels(templates, chans_and_nbrs):
    """Assign templates to their maximum-amplitude channels. Subset templates by selected channels. 



    Args:
        templates: A 3-D numpy array of shape (n_templates, n_channels, n_timepoints).
            This is typically the template waveforms across the entire eletrode array. 
        channel_and_nbrs: A dict mapping of selected channels to their neighbors.  
    Return:
        idx_templates: An array of template ids that belong to (peaks at) one of the selected channels. 
        selected_chans: A list of numpy arrays, in which each element is an numpy array of 
            selected channels corresponding to the template ids. 
        chan_to_template: A dict mapping channel IDs to templates that belong to each selected channel. 
    """
    channels = np.array(list(chans_and_nbrs.keys()))  # targselectedet channels 

    # the channels with max amplitude for each template
    max_chans = templates.ptp(2).argmax(1)
    print("max chans: ", max_chans.shape)

    # find all template units on selected channels
    idx_templates = np.where(np.isin(max_chans, channels))[0]
    print("templates in selected chanels {}".format(len(idx_templates)))

    max_chans_selected = max_chans[idx_templates]
    if len(set([len(chans_and_nbrs[ch]) for ch in max_chans_selected])) == 1:
        selected_chans = np.stack([chans_and_nbrs[ch]
                                   for ch in max_chans_selected])
        print("selectecd chans for all templates: ", selected_chans.shape)
    else:
        selected_chans = [chans_and_nbrs[ch] for ch in max_chans_selected]

    center_channels = [x[0] for x in selected_chans]

    chan_to_template = defaultdict(list)
    for ch, temp_id in zip(center_channels, idx_templates):
        chan_to_template[ch].append(temp_id)

    return idx_templates, selected_chans, chan_to_template


def create_upsampled_templates(templates, idx_templates, selected_chans, upsample=10):
    """Upsample, shift and downsamples templates using scipy.signal.resample. 

    First upsample the template by upsample times, then shift each upsampled templates by 
    1/upsample, 2/upsample, 3/upsample... finally downsample to the original time resolution. 

    Args:
        templates: A 3-D numpy array of shape (n_templates, n_channels, n_timepoints) 
            representing all template waveforms. 
        idx_templates: A numpy array of selected template IDs
        selected_chans: A list of numpy arrays, in which each element is an numpy array of 
            selected channels corresponding to the template ids. 
        upsample: integer Factor of upsampling. 
    Returns:
        templates_resampled: A 4-D numpy array of upsampled, shifted and then downsampled templates. 
    """

    templates_resampled = resample(
        templates, templates.shape[2] * upsample, axis=2, window=0)
    n_nbrs = selected_chans.shape[1]
    idx = []
    for k in range(upsample):  # 0,1,2,...
        idx.append(np.arange(k, upsample * templates.shape[2], upsample))
    idx = np.array(idx)

    templates_resampled = templates_resampled[:, :, idx]

    templates_resampled = templates_resampled[np.tile(
        idx_templates, (n_nbrs, 1)).T, selected_chans]
    templates_resampled = templates_resampled.swapaxes(2, 3)
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
    return spike_time_labels[np.isin(spike_time_labels[:, 1], template_ids)]


def subset_spike_time_by_channel(spike_time_labels, channel_id):
    """
    Args:
        spike_time_labels: [n_spikes, 2] each row (time, channel_id)
        channel_id: int, channel_id
    Return:
        a subset of spike_time_labels from a particular channel.
    """
    return spike_time_labels[spike_time_labels[:, 1] == channel_id]


def template_window(templates, n_timesteps, offset):
    time_length = templates.shape[1]
    start = (time_length - n_timesteps) // 2 + \
        offset  # take the middle chunk with offset
    end = start + n_timesteps
    templates = templates[:, start:end, :]
    return templates
