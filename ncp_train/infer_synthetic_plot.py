 
import numpy as np
import torch
import time
import json 
import argparse
import sys, os 
sys.path.append("../spike_sorting")
from ncp_sampler import NCP_Sampler
from spike_data_generator import *
from spike_utils import *


parser = argparse.ArgumentParser(description='Plot inference results.')
parser.add_argument('infer_dir', type=str)
#parser.add_argument('--templates', dest="templates_name", type=str, default="ks2-190430")
parser.add_argument('--min_cls_size', type=int, default=0)
parser.add_argument('--topn', type=int, default=1)
parser.add_argument('--plot_mfm', action="store_const", const=True, default=False)
parser.add_argument('--plot_type', type=str, default="overlay")

# python infer_49ch_non_triaged_plot.py --min_cls_size 50s --plot_mfm --plot_type overlay

args = parser.parse_args()


do_corner_padding = True 

output_dir = args.infer_dir
with open(os.path.join(output_dir, "infer_params.json"), "r") as f:
    infer_params = json.load(f)

# inference on 49 ch dataset 
min_cls_size = args.min_cls_size
# templates_name = args.templates_name

templates = None 
templates_use = None 
templates_name = None
infer_params['nbr_dist'] = 70
infer_params['n_nbr'] = 7
# infer_params['template_offset'] = 0
# if templates_name == "yass-190322":
#     infer_params['template_file'] = "/media/yueqi/Data1/data_ml/spike_sorting/data_49ch/templates_cluster_data1_allset_49chan.npy"
#     templates = np.load(infer_params['yass_template_file']).transpose(0,2,1)
#     # template: (151, 61, 49) [n_templates, n_times, n_channels]
#     # transpose to (151, 49, 61) [n_templates, n_channels, n_times]
# elif templates_name == "ks2-190430":
#     infer_params['template_file'] = "/home/yueqi/Dropbox/lib/discrete_neural_process/data/data_190430_non-triaged_and_kilosort/kilosort2/templates_reloaded.npy"
#     templates = np.load(infer_params['template_file']).transpose(2,1,0)
#     # template: (61, 49, 143) [n_times, n_channels, n_templates]
#     # transpose to (151, 49, 61) [n_templates, n_channels, n_times]
#     infer_params['template_offset'] = 4
# else:
#     raise ValueError("unknown template name")

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

# idx_templates, selected_chans, chan_to_template = select_template_channels(templates, chan_to_nbrs)

# templates_use = templates.transpose(0,2,1)
# (151, 61, 49) [n_templates, n_times, n_channels]
# templates_use = template_window(templates_use, infer_params['n_timesteps'], infer_params['template_offset'])


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
            clusters, nll, data_arr, geom, nbr_channels, default_colors, topn=topn, 
            extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
            min_cls_size=min_cls_size, templates=temp_in_ch, template_name=templates_name,
            figdir=fig_dir_overlay, fname_postfix=fname, size_single=(9,6), 
            plot_params={"time_scale":1.1, "scale":8., "alpha_overlay":0.1})
        
        n_ch = len(nbr_channels)
        vertical_geom = np.stack([np.zeros(n_ch), - np.arange(n_ch) * 12 * 7]).T
        
        plot_spike_clusters_and_templates_overlay(
            clusters, nll, data_arr, vertical_geom, np.arange(n_ch), default_colors, topn=topn, 
            extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
            min_cls_size=min_cls_size, templates=temp_in_ch, template_name=templates_name,
            figdir=fig_dir_vert_overlay, fname_postfix=fname, size_single=(2.5,18), vertical=True,
            plot_params={"time_scale":1.1, "scale":8., "alpha_overlay":0.1})
    
    elif args.plot_type == 'tsne':
        fig_dir_tsne = os.path.join(output_dir, "figs_tsne_min-cls-{}".format(min_cls_size))
        if not os.path.isdir(fig_dir_tsne): os.mkdir(fig_dir_tsne)
        tsne_dir = os.path.join(infer_params['data_name'], "spike_encoder_it-18600/data_encoder")
        fname_str = "ch-{}_seed-{}".format(ch, seed)
        fname = [x for x in os.listdir(tsne_dir) if fname in x and x.endswith(".npz")]
        data_encoded = np.load(os.path.join(tsne_dir, "{}".format(fname[0])))
        data_encoded = data_encoded['encoded_spikes']
        fname = fname[0].rstrip("_encoded_spikes.npz")
        plot_raw_and_encoded_spikes_tsne(
            clusters, nll, data_arr, data_encoded, default_colors, topn=topn, 
            extra_clusters=mfm_clusters, extra_name=mfm_name, gt_labels=gt_labels, 
            min_cls_size=min_cls_size, sort_by_count=True,
            figdir=fig_dir_tsne, fname_postfix=fname, size_single=(6,6),
            tsne_params={'seed': 0, 'perplexity': 30},
            plot_params={'pt_scale': 1}, show=False
        )
