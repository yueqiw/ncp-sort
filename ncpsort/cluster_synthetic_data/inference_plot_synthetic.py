 
"""Plot clustered spikes
Usage:
    python ncpsort.cluster_synthetic_data.inference_plot_synthetic \
        --inference_dir ./inference_synthetic_N-1000/cluster_S-150-beam_NCP-10000 \
        --min_cls_size 50 --plot_type overlay

    or  --inference_dir --min_cls_size 50 --plot_type tsne
"""

import numpy as np
import torch
import time
import json 
import argparse
import os 
from ncpsort.utils.spike_utils import get_chan_nbrs, select_template_channels, template_window
from ncpsort.utils.plotting import DEFAULT_COLORS
from ncpsort.utils.plotting import plot_spike_clusters_and_gt_in_rows
from ncpsort.utils.plotting import plot_spike_clusters_and_templates_overlay
from ncpsort.utils.plotting import plot_raw_and_encoded_spikes_tsne


parser = argparse.ArgumentParser(description='Plot inference results.')
parser.add_argument('--inference_dir', type=str)
parser.add_argument('--min_cls_size', type=int, default=0)
parser.add_argument('--topn', type=int, default=1)
parser.add_argument('--plot_mfm', action="store_const", const=True, default=False)
parser.add_argument('--plot_type', type=str, default="overlay")


if __name__ == "__main__":
    args = parser.parse_args()

    do_corner_padding = True 

    output_dir = args.inference_dir
    with open(os.path.join(output_dir, "infer_params.json"), "r") as f:
        infer_params = json.load(f)

    min_cls_size = args.min_cls_size

    templates = None 
    templates_use = None 
    templates_name = None
    infer_params['nbr_dist'] = 70
    infer_params['n_nbr'] = 7

    print("parameters:\n", json.dumps(infer_params, indent=2))

    geom = np.array([
        [-585.0, 270.0],
        [-645.0, 270.0],
        [-525.0, 270.0],
        [-615.0, 210.0],
        [-555.0, 210.0],
        [-615.0, 330.0],
        [-555.0, 330.0]]
    )
    chans_with_nbrs, chan_to_nbrs = get_chan_nbrs(geom, infer_params['nbr_dist'], infer_params['n_nbr'], keep_less_nbrs=False)
    print("{} channels used:".format(len(chans_with_nbrs)))
    print(chans_with_nbrs)

    topn = args.topn

    data_dir = os.path.join(output_dir, "data_ncp")
    # fig_dir_by_row = os.path.join(output_dir, "figures_by_row")
    # if not os.path.isdir(fig_dir_by_row): os.mkdir(fig_dir_by_row)
    fig_dir_overlay = os.path.join(output_dir, "figs_overlay_min-cls-{}_temp-{}".format(min_cls_size, templates_name))
    if not os.path.isdir(fig_dir_overlay): os.mkdir(fig_dir_overlay)
    fig_dir_vert_overlay = os.path.join(output_dir, "figs_overlay_vertical_min-cls-{}_temp-{}".format(min_cls_size, templates_name))
    if not os.path.isdir(fig_dir_vert_overlay): os.mkdir(fig_dir_vert_overlay)

    if args.plot_mfm:
        mfm_dir = os.path.join(infer_params['data_name'], "cluster_mfm", "data_mfm")

    input_dir = infer_params['data_name']
    fnames_list = [x.rstrip(".npz") for x in os.listdir(os.path.join(input_dir, "data_input")) if x.endswith(".npz")]
    fnames_list = sorted(fnames_list)

    for fname in fnames_list:

        if args.plot_mfm:
            mfm_fname = [x for x in os.listdir(mfm_dir) if fname in x and x.endswith(".npy")]
            mfm_fname = mfm_fname[0].rstrip(".npy")
            npy_fname = os.path.join(mfm_dir, "{}.npy".format(mfm_fname))
            mfm_clusters = np.load(npy_fname)
            mfm_name = "MFM"
        else:
            mfm_clusters = None
            mfm_name = None

        print("Plotting {}:".format(fname))
        npz_fname = os.path.join(data_dir, "{}_ncp.npz".format(fname))
        npz = np.load(npz_fname)
        clusters, nll, data_arr, gt_labels = npz['clusters'], npz['nll'], npz['data_arr'], npz['gt_labels']

        # plot_spike_clusters_and_gt_in_rows(
        #         css, nll, data_arr, gt_labels, topn=topn, 
        #         figdir=fig_dir_by_row, fname_postfix=fname,
        #         plot_params={"spacing":1.25, "width":0.9, "vscale":1.5, "subplot_adj":0.9}, 
        #         downsample=3)
        temp_in_ch = None 
        templates_name = "{} templates".format(templates_name) if templates_name else None 
        nbr_channels = np.arange(len(geom))

        if args.plot_type == 'overlay':
            plot_spike_clusters_and_templates_overlay(
                clusters, nll, data_arr, geom, nbr_channels, DEFAULT_COLORS, topn=topn, 
                extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
                min_cls_size=min_cls_size, templates=temp_in_ch, template_name=templates_name,
                figdir=fig_dir_overlay, fname_postfix=fname, size_single=(9,6), 
                plot_params={"time_scale":1.1, "scale":8., "alpha_overlay":0.1})
            
            n_ch = len(nbr_channels)
            vertical_geom = np.stack([np.zeros(n_ch), - np.arange(n_ch) * 12 * 7]).T
            
            plot_spike_clusters_and_templates_overlay(
                clusters, nll, data_arr, vertical_geom, np.arange(n_ch), DEFAULT_COLORS, topn=topn, 
                extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
                min_cls_size=min_cls_size, templates=temp_in_ch, template_name=templates_name,
                figdir=fig_dir_vert_overlay, fname_postfix=fname, size_single=(2.5,18), vertical=True,
                plot_params={"time_scale":1.1, "scale":8., "alpha_overlay":0.1})
        
        elif args.plot_type == 'tsne':
            fig_dir_tsne = os.path.join(output_dir, "figs_tsne_min-cls-{}".format(min_cls_size))
            if not os.path.isdir(fig_dir_tsne): os.mkdir(fig_dir_tsne)
            tsne_dir = os.path.join(infer_params['data_name'], "spike_encoder_it-18600/data_encoder")
            fname = [x for x in os.listdir(tsne_dir) if fname in x and x.endswith(".npz")]
            data_encoded = np.load(os.path.join(tsne_dir, "{}".format(fname[0])))
            data_encoded = data_encoded['encoded_spikes']
            fname = fname[0].rstrip("_encoded_spikes.npz")
            plot_raw_and_encoded_spikes_tsne(
                clusters, nll, data_arr, data_encoded, DEFAULT_COLORS, topn=topn, 
                extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
                min_cls_size=min_cls_size, sort_by_count=True,
                figdir=fig_dir_tsne, fname_postfix=fname, size_single=(6,6),
                tsne_params={'seed': 0, 'perplexity': 30},
                plot_params={'pt_scale': 1}, show=False
            )
