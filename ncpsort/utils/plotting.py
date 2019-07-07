import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from ncpsort.utils.clustering import get_topn_clusters

DEFAULT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9', 'C7',
                  'black', 'blue', 'red', 'green', 'magenta', 'brown', 'orange']


def plot_raw_spikes_in_rows(waveforms, assignments, spacing=1, width=1, vscale=1, 
                            subplot_adj=0.9, colors=DEFAULT_COLORS, figtitle="", 
                            figdir="./", fname_postfix="", show=True):
    """Plot raw spikes, each spike in a separate row and each channel in a separate column
    
    Args:
        waveforms: a numpy array of shape (n_samples, n_timesteps, n_channels)
        assignments: a numpy array of shape (n_samples,)
    """
    waveforms = waveforms.transpose((0, 2, 1))  # [N, n_chs, n_times]
    n_samples = waveforms.shape[0]
    n_chs = waveforms.shape[1]
    n_unit = len(set(assignments))

    waveforms_plot = waveforms * vscale - \
        np.reshape(np.arange(n_samples), (-1, 1, 1)) * spacing * 8

    fig_height = 1
    fig, axes = plt.subplots(1, n_chs, figsize=(
        width * n_chs, 2 + (n_samples - 1) * spacing / 4), sharey=True)
    fontsize = 15
    
    # plt.ylim(np.percentile(waveforms, 0.1), np.percentile(waveforms, 99.9))
    plt.ylim(np.min(waveforms_plot) - 2, np.max(waveforms_plot) + 2)

    units = np.unique(assignments[assignments != -1])
    for chid in range(n_chs):
        for unit in units:
            axes[chid].plot(waveforms_plot[assignments == unit, chid, :].T,
                            color=DEFAULT_COLORS[unit % 20], alpha=0.8, label="unit {}".format(unit))
        if np.sum(assignments == -1) > 0:
            axes[chid].plot(waveforms_plot[assignments == -1, chid, :].T,
                            color='gray', alpha=0.8, label="unlabeled")

    for chid in range(n_chs):
        axes[chid].set_title("CH {}".format(chid), fontsize=fontsize)
        axes[chid].set_axis_off()

    fig.suptitle(figtitle, fontsize=fontsize+13,
                 y=1 - (1-0.99) * 100 / n_samples)
    plt.tight_layout()
    plt.subplots_adjust(top=1 - (1-subplot_adj) * 100 / n_samples)
    if show:
        plt.show()
        return None
    else:
        save_path = os.path.join(
            figdir, "sample_{}_{}.png".format(n_samples, fname_postfix))
        plt.savefig(save_path)
        plt.close()
        return save_path


def plot_raw_spikes_overlay(waveforms, assignments, geom, channels, colors, sort_by_count=True,
                            min_cls_size=10,
                            time_scale=3., scale=10., alpha_overlay=0.2, alpha_single=0.8,
                            figtitle="", titlesize=25, size_single=(9, 9), vertical=False,
                            figdir="./", fname_postfix="", show=True):
    """Plot spikes, with all spikes overlayed on each other, and spatially arranged according to geom

    Args:
        waveforms: a numpy array of shape (n_samples, n_timesteps, n_channels)
        assignments: a numpy array of shape (n_samples,) of cluster assignments
        geom: a numpy array of shape (n_channels, 2), spatial locations for each channel
        channels: array of shape (n_channels,), list of channel_id correspond to the channel axis of waveforms
    """
    n_samples, n_timesteps, n_channels = waveforms.shape

    clusters, cls_count = np.unique(assignments, return_counts=True)
    if min_cls_size is not None and min_cls_size > 0:
        clusters = clusters[cls_count > min_cls_size]
        cls_count = cls_count[cls_count > min_cls_size]

    if sort_by_count:
        clusters = clusters[np.argsort(-cls_count)]  # sort by count
    else:
        clusters = np.sort(clusters)

    n_clusters = len(clusters)
    if vertical:
        fig, axes = plt.subplots(1, n_clusters * 2, figsize=(2 * n_clusters * size_single[0], size_single[1]),
                                 sharex=True, sharey=True)
        axes = axes.reshape((-1, 2))
    else:
        fig, axes = plt.subplots(n_clusters, 2, figsize=(2 * size_single[0], n_clusters * size_single[1]),
                                 sharex=True, sharey=True)
        if n_clusters == 1:
            axes = [axes]

    geom_use = geom[channels]
    center = geom_use[0]

    for i, cl in enumerate(clusters):
        wave_subset = waveforms[assignments == cl]
        count = np.sum(assignments == cl)

        time_axis = np.tile(np.arange(n_timesteps), len(wave_subset)).reshape(
            (-1, n_timesteps))[:, :, np.newaxis]
        plot_x = geom_use[:, 0] + (time_axis - n_timesteps/2) / time_scale
        plot_y = geom_use[:, 1] + wave_subset * scale
        plot_x = plot_x.transpose(1, 0, 2).reshape(n_timesteps, -1)
        plot_y = plot_y.transpose(1, 0, 2).reshape(n_timesteps, -1)
        axes[i][0].plot(plot_x, plot_y, color=colors[i], alpha=alpha_overlay)
        if not vertical:
            axes[i][0].text(geom_use[:, 0].max(), geom_use[:,
                                                           1].max(), "n = {}".format(count), fontsize=20)
        else:
            axes[i][0].set_title("n = {}".format(count), fontsize=titlesize)
            axes[i][1].set_title("Averaged", fontsize=titlesize)

        wave_subset_avg = wave_subset.mean(0)
        wave_subset_std = wave_subset.std(0)
        time_axis = np.arange(n_timesteps)[:, np.newaxis]
        plot_x = geom_use[:, 0] + (time_axis - n_timesteps/2) / time_scale
        plot_y = geom_use[:, 1] + wave_subset_avg * scale
        axes[i][1].plot(plot_x, plot_y, color=colors[i],
                        alpha=alpha_single, linewidth=3)

        geom_y_range = max(geom_use[:, 1].max(
        ) - center[1], center[1] - geom_use[:, 1].min()) * 2
        geom_x_range = max(geom_use[:, 0].max(
        ) - center[0], center[0] - geom_use[:, 0].min()) * 2
        if not vertical:
            y_high = center[1] + geom_y_range
            y_low = center[1] - geom_y_range
            x_high = center[0] + geom_x_range / 2 + geom_x_range / 6
            x_low = center[0] - geom_x_range / 2 - geom_x_range / 6
            for xi in [0, 1]:
                axes[i][xi].set_xlim(x_low, x_high)
        else:
            y_high = center[1] + geom_y_range / 2 + geom_y_range / 6
            y_low = center[1] - geom_y_range / 2 - geom_y_range / 6
        for xi in [0, 1]:
            axes[i][xi].set_ylim(y_low, y_high)
            axes[i][xi].set_axis_off()

        fill_x = (np.arange(n_timesteps) - n_timesteps/2) / time_scale
        # fill_y1 = -np.ones(fill_x.shape) * scale
        # fill_y2 = np.ones(fill_x.shape) * scale

        for k in range(n_channels):
            for col in [0, 1]:
                if not vertical:
                    axes[i][col].text(geom_use[k, 0] + n_timesteps / 2 / time_scale + 1,
                                      geom_use[k, 1] + 5, str(channels[k]), fontsize=15)
                axes[i][col].fill_between(fill_x + geom_use[k, 0],
                                          wave_subset_std[:, k] *
                                          scale + plot_y[:, k],
                                          -wave_subset_std[:, k] *
                                          scale + plot_y[:, k],
                                          color='grey', alpha=0.1)
    if vertical:
        for k in range(n_channels):
            axes[0][0].text(geom_use[k, 0] - n_timesteps / 2 / time_scale - 6,
                            geom_use[k, 1] + 5, str(channels[k]), fontsize=15)
        fig.suptitle(figtitle, fontsize=titlesize)
    else:
        axes[0][0].set_title(figtitle + " (Raw)", fontsize=titlesize)
        axes[0][1].set_title(figtitle + " (Averaged)", fontsize=titlesize)

    plt.tight_layout()
    if vertical:
        plt.subplots_adjust(top=0.9)
    if show:
        plt.show()
        return None
    else:
        save_path = os.path.join(figdir, "{}.png".format(fname_postfix))
        plt.savefig(save_path)
        plt.close()
        return save_path


def plot_templates_separate(templates, geom, channels, colors,
                            time_scale=3., scale=10., alpha_overlay=0.2, alpha_single=0.8,
                            figtitle="", titlesize=25, size_single=(9, 9), vertical=False,
                            figdir="./", fname_postfix="", show=True):
    """Plot a set of templates 

    Args:
        templates: a numpy array of shape (n_templates, n_timesteps, n_channels)
        geom: a numpy array of shape (n_channels, 2), spatial locations for each channel
        channels: array of shape (n_channels,), list of channel_id correspond to the channel axis of waveforms
    """
    n_templates, n_timesteps, n_channels = templates.shape
    if vertical:
        fig, axes = plt.subplots(1, n_templates, figsize=(n_templates * size_single[0], 1 * size_single[1]),
                                 sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(n_templates, 1, figsize=(1 * size_single[0], n_templates * size_single[1]),
                                 sharex=True, sharey=True)
    if n_templates == 1:
        axes = [axes]
    geom_use = geom[channels]
    center = geom_use[0]

    for i in np.arange(n_templates):
        template_single = templates[i]

        time_axis = np.arange(n_timesteps)[:, np.newaxis]
        plot_x = geom_use[:, 0] + (time_axis - n_timesteps/2) / time_scale
        plot_y = geom_use[:, 1] + template_single * scale
        axes[i].plot(plot_x, plot_y, color=colors[i],
                     alpha=alpha_single, linewidth=3)

        geom_y_range = max(geom_use[:, 1].max(
        ) - center[1], center[1] - geom_use[:, 1].min()) * 2
        geom_x_range = max(geom_use[:, 0].max(
        ) - center[0], center[0] - geom_use[:, 0].min()) * 2
        if not vertical:
            y_high = center[1] + geom_y_range
            y_low = center[1] - geom_y_range
            x_high = center[0] + geom_x_range / 2 + geom_x_range / 6
            x_low = center[0] - geom_x_range / 2 - geom_x_range / 6
            for xi in [0, 1]:
                axes[i].set_xlim(x_low, x_high)
        else:
            y_high = center[1] + geom_y_range / 2 + geom_y_range / 6
            y_low = center[1] - geom_y_range / 2 - geom_y_range / 6
        for xi in [0, 1]:
            axes[i].set_ylim(y_low, y_high)
            axes[i].set_axis_off()

        # fill_x = (np.arange(n_timesteps) - n_timesteps/2) / time_scale
        # fill_y1 = -np.ones(fill_x.shape) * scale
        # fill_y2 = np.ones(fill_x.shape) * scale

        for k in range(n_channels):
            if not vertical:
                axes[i].text(geom_use[k, 0] + n_timesteps / 2 / time_scale + 1,
                             geom_use[k, 1] + 5, str(channels[k]), fontsize=15)
            # axes[i].fill_between(fill_x + geom_use[k,0], fill_y2 + geom_use[k,1], fill_y1 + geom_use[k,1], color='grey', alpha=0.1)

    if vertical:
        for k in range(n_channels):
            axes[0].text(geom_use[k, 0] - n_timesteps / 2 / time_scale - 6,
                         geom_use[k, 1] + 5, str(channels[k]), fontsize=15)
        fig.suptitle(figtitle, fontsize=titlesize)
    else:
        axes[0].set_title(figtitle, fontsize=titlesize)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    else:
        save_path = os.path.join(
            figdir, "{}_templates_{}.png".format(fname_postfix, n_templates))
        plt.savefig(save_path)
        plt.close()
        return save_path


def plot_spike_clusters_and_gt_in_rows(clusters, nll, data_arr, gt_labels, topn=2, figdir="./", fname_postfix="",
                                       plot_params={"spacing": 1.25, "width": 0.9, "vscale": 1.5, "subplot_adj": 0.9}, downsample=None):
    """Plot spikes colored by assigned clusters vs. ground truth clusters, each spike as a row
    """
    topn_clusters, topn_nll = get_topn_clusters(clusters, nll, topn)

    reorder = np.argsort(gt_labels)
    gt_labels = gt_labels[reorder]
    data_arr = data_arr[reorder]
    topn_clusters = topn_clusters[:, reorder]

    gt_path = plot_raw_spikes_in_rows(data_arr, gt_labels,
                                      spacing=plot_params["spacing"], width=plot_params["width"],
                                      vscale=plot_params["vscale"],
                                      subplot_adj=plot_params['subplot_adj'],
                                      figtitle='Ground truth', figdir=figdir, fname_postfix=fname_postfix + '_rows_gt', show=False)
    fig_paths = [gt_path]

    for i in range(len(topn_nll)):
        snll = topn_nll[i]
        cs = topn_clusters[i]

        K = len(set(cs))
        pr = np.exp(-snll)
        title = 'NCP: {} Clusters (Prob: {:.3f})'.format(K, pr)
        fpath = plot_raw_spikes_in_rows(data_arr, cs,
                                        spacing=plot_params["spacing"], width=plot_params["width"],
                                        vscale=plot_params["vscale"],
                                        subplot_adj=plot_params['subplot_adj'],
                                        figtitle=title, figdir=figdir, fname_postfix=fname_postfix + '_rows_pred{}'.format(i), show=False)
        fig_paths.append(fpath)
    combine_imgs(fig_paths, fpath.replace(
        "pred", "pred"), downsample=downsample)
    for f in fig_paths:
        if os.path.exists(f):
            os.remove(f)


def plot_spike_clusters_and_gt_overlay(clusters, nll, data_arr, gt_labels, geom, channels, colors, topn=2,
                                       sort_by_count=True, min_cls_size=0,
                                       figdir="./", fname_postfix="", size_single=(9, 9), vertical=False,
                                       plot_params={"time_scale": 1.1, "scale": 8., "alpha_overlay": 0.1}):
    """Plot spikes colored by assigned clusters vs. ground truth clusters, with all spikes overlayed on each other
    """
    gt_path = plot_raw_spikes_overlay(data_arr, gt_labels, geom, channels, colors, size_single=size_single,
                                      vertical=vertical,
                                      time_scale=plot_params['time_scale'],
                                      scale=plot_params['scale'],
                                      alpha_overlay=plot_params['alpha_overlay'],
                                      figtitle="Ground Truth", titlesize=25,
                                      figdir=figdir, fname_postfix=fname_postfix + '_overlay_gt', show=False)
    fig_paths = [gt_path]

    topn_clusters, topn_nll = get_topn_clusters(clusters, nll, topn)

    for i in range(topn):
        snll = topn_nll[i]
        cs = topn_clusters[i]
        K = len(set(cs))
        pr = np.exp(-snll)
        title = 'NCP: {} Clusters (Prob: {:.3f})'.format(K, pr)
        topn_path = plot_raw_spikes_overlay(data_arr, cs, geom, channels, colors, size_single=size_single,
                                            vertical=vertical,
                                            min_cls_size=min_cls_size,
                                            sort_by_count=sort_by_count,
                                            time_scale=plot_params['time_scale'],
                                            scale=plot_params['scale'],
                                            alpha_overlay=plot_params['alpha_overlay'],
                                            figtitle=title, titlesize=25,
                                            figdir=figdir, fname_postfix=fname_postfix + '_overlay_top{}'.format(i+1), show=False)
        fig_paths.append(topn_path)

    if vertical:
        combine_imgs_vertical(fig_paths, gt_path.replace("_gt", "_combined"))
    else:
        combine_imgs(fig_paths, gt_path.replace("_gt", "_combined"))
    for f in fig_paths:
        if os.path.exists(f):
            os.remove(f)


def plot_spike_clusters_and_templates_overlay(clusters, nll, data_arr, geom, channels, colors,
                                              topn=1, min_cls_size=10,
                                              templates=None, template_name="Templates",
                                              gt_labels=None,
                                              extra_clusters=None, extra_name=None,
                                              sort_by_count=True, figdir="./", fname_postfix="", size_single=(9, 9), vertical=False,
                                              plot_params={"time_scale": 1.1, "scale": 8., "alpha_overlay": 0.1}):
    """Plot spikes colored by assigned clusters and their templates, with all spikes overlayed on each other
    """
    fig_paths = []

    if gt_labels is not None:
        K = len(set(gt_labels))
        title = 'Groud Truth: {} Clusters'.format(K)
        extra_path = plot_raw_spikes_overlay(data_arr, gt_labels, geom, channels, colors, size_single=size_single,
                                             vertical=vertical,
                                             min_cls_size=min_cls_size,
                                             sort_by_count=sort_by_count,
                                             time_scale=plot_params['time_scale'],
                                             scale=plot_params['scale'],
                                             alpha_overlay=plot_params['alpha_overlay'],
                                             figtitle=title, titlesize=25,
                                             figdir=figdir, fname_postfix=fname_postfix + '_overlay_gt', show=False)
        fig_paths.append(extra_path)

    if templates is not None:
        gt_path = plot_templates_separate(templates, geom, channels, colors, size_single=size_single,
                                          vertical=vertical,
                                          time_scale=plot_params['time_scale'],
                                          scale=plot_params['scale'],
                                          alpha_overlay=1,
                                          figtitle=template_name, titlesize=25,
                                          figdir=figdir, fname_postfix=fname_postfix, show=False)
        fig_paths.append(gt_path)

    topn_clusters, topn_nll = get_topn_clusters(clusters, nll, topn)

    for i in range(topn):
        snll = topn_nll[i]
        cs = topn_clusters[i]
        K = len(set(cs))
        pr = np.exp(-snll)
        title = 'NCP: {} Clusters (Prob: {:.3f})'.format(K, pr)
        topn_path = plot_raw_spikes_overlay(data_arr, cs, geom, channels, colors, size_single=size_single,
                                            vertical=vertical,
                                            min_cls_size=min_cls_size,
                                            sort_by_count=sort_by_count,
                                            time_scale=plot_params['time_scale'],
                                            scale=plot_params['scale'],
                                            alpha_overlay=plot_params['alpha_overlay'],
                                            figtitle=title, titlesize=25,
                                            figdir=figdir, fname_postfix=fname_postfix + '_overlay_pred{}'.format(i+1), show=False)
        fig_paths.append(topn_path)

    if extra_clusters is not None and extra_name is not None:
        K = len(set(extra_clusters))
        title = '{}: {} Clusters'.format(extra_name, K)
        extra_path = plot_raw_spikes_overlay(data_arr, extra_clusters, geom, channels, colors, size_single=size_single,
                                             vertical=vertical,
                                             min_cls_size=min_cls_size,
                                             sort_by_count=sort_by_count,
                                             time_scale=plot_params['time_scale'],
                                             scale=plot_params['scale'],
                                             alpha_overlay=plot_params['alpha_overlay'],
                                             figtitle=title, titlesize=25,
                                             figdir=figdir, fname_postfix=fname_postfix + '_overlay_{}'.format(extra_name), show=False)
        fig_paths.append(extra_path)

    if vertical:
        combine_imgs_vertical(fig_paths, topn_path.replace("pred", "top"))
    else:
        combine_imgs(fig_paths, topn_path.replace("pred", "top"))

    for f in fig_paths:
        if os.path.exists(f):
            os.remove(f)


def plot_raw_and_encoded_spikes_tsne(clusters, nll, data_arr, data_encoded, colors, topn=2,
                                     extra_clusters=None, extra_name=None, gt_labels=None,
                                     min_cls_size=10, sort_by_count=True,
                                     figdir="./", fname_postfix="", size_single=(6, 6),
                                     tsne_params={'seed': 0, 'perplexity': 30},
                                     plot_params={'pt_scale': 1}, show=True):
    """Plot spike waveforms and the NCP-encoded vectors using t-SNE
    """

    from MulticoreTSNE import MulticoreTSNE as TSNE

    N = data_arr.shape[0]
    pt_size = 26.*(100/N)**0.5 * plot_params['pt_scale']
    topn_clusters, topn_nll = get_topn_clusters(clusters, nll, topn)

    n_plots = topn
    plot_clusters = topn_clusters[:topn]
    plot_clusters_names = ['NCP clusters #{}'.format(i) for i in range(topn)]

    if extra_clusters is not None and extra_name is not None:
        n_plots += 1
        plot_clusters = np.concatenate(
            [topn_clusters[:topn], [extra_clusters]])
        plot_clusters_names.append(extra_name)

    if gt_labels is not None:
        n_plots += 1
        plot_clusters = np.concatenate([[gt_labels], topn_clusters[:topn]])
        plot_clusters_names = ["Ground Truth"] + plot_clusters_names

    fig, axes = plt.subplots(n_plots, 2, figsize=(
        size_single[0] * 2, size_single[1] * n_plots))
    if n_plots == 1:
        axes = [axes]
    tsne = TSNE(n_jobs=4, n_components=2,
                perplexity=tsne_params['perplexity'], random_state=tsne_params['seed'])
    tsne_encoded = tsne.fit_transform(data_encoded)
    data_reshape = data_arr.reshape((data_arr.shape[0], -1))
    tsne_raw = tsne.fit_transform(data_reshape)

    for i in range(n_plots):
        cs = plot_clusters[i]
        n_clusters = len(set(cs))

        cluster_ids, counts = np.unique(cs, return_counts=True)
        if sort_by_count:
            sorted_idx = np.argsort(-counts)
            cluster_ids, counts = cluster_ids[sorted_idx], counts[sorted_idx]
            cluster_rename = {cluster_ids[i]                              : i for i in range(len(cluster_ids))}
            cluster_ids = np.vectorize(cluster_rename.get)(cluster_ids)
            clusters_use = np.vectorize(cluster_rename.get)(cs)
        else:
            clusters_use = cs

        large_clusters = cluster_ids[counts > min_cls_size]
        small_clusters = cluster_ids[counts <= min_cls_size]
        color_mapping = {cluster_ids[i]: (colors[i] if cluster_ids[i] in large_clusters else 'grey')
                         for i in range(len(cluster_ids))}

        fontsize = 12
        for k in cluster_ids:
            mask = (clusters_use == k)
            axes[i][0].scatter(tsne_raw[mask, 0], tsne_raw[mask, 1],
                               color=color_mapping[k], s=pt_size)
            axes[i][1].scatter(
                tsne_encoded[mask, 0], tsne_encoded[mask, 1], color=color_mapping[k], s=pt_size)

        axes[i][0].set_title(
            't-SNE of raw spikes ({})'.format(plot_clusters_names[i]), fontsize=fontsize)
        axes[i][1].set_title(
            't-SNE of NCP-encoded spikes ({})'.format(plot_clusters_names[i]), fontsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    else:
        save_path = os.path.join(figdir, "{}_tsne.png".format(fname_postfix))
        plt.savefig(save_path)
        save_path = os.path.join(figdir, "{}_tsne.pdf".format(fname_postfix))
        plt.savefig(save_path)
        plt.close()
        return save_path


def plot_samples_tsne(model, data_generator, sampler, N=50, seed=None, save_name=None, s=26):
    """Plot data produced from a generator and the cluster labels using t-SNE 
    """
    from MulticoreTSNE import MulticoreTSNE as TSNE

    if seed:
        np.random.seed(seed=seed)

    data, cs, clusters, num_clusters = data_generator.generate(N, batch_size=1)

    tsne = TSNE(n_jobs=4, n_components=2, perplexity=30.0, random_state=seed)
    data_tsne = tsne.fit_transform(data[0])

    fig, ax = plt.subplots(ncols=7, nrows=1, num=1)
    ax = ax.reshape(7)

    N = data.shape[1]
    # s = 26  #size for scatter
    fontsize = 15

    ax[0].scatter(data_tsne[:, 0], data_tsne[:, 1], color='gray', s=s)

    K = len(set(cs))

    ax[0].set_title(str(N) + ' Points', fontsize=fontsize)

    for j in range(N):
        xs = data_tsne[j, 0]
        ys = data_tsne[j, 1]
        ax[1].scatter(xs, ys, color='C'+str((cs[j]+1) % 10), s=s)

    ax[1].set_title('Ground Truth: {} Clusters'.format(K), fontsize=fontsize)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(2)
        ax[1].spines[axis].set_linewidth(2)

    ncp_sampler = sampler(model, data)
    S = 5000
    css, nll = ncp_sampler.sample(S)

    sorted_nll = np.sort(list(set(nll)))

    for i in range(5):  # top 5 by nll
        ax[i+2].cla()
        snll = sorted_nll[i]
        r = np.nonzero(nll == snll)[0][0]
        cs = css[r, :]

        for j in range(N):
            xs = data_tsne[j, 0]
            ys = data_tsne[j, 1]
            ax[i+2].scatter(xs, ys, color='C'+str((cs[j]+1) % 10), s=s)

        K = len(set(cs))
        prob = np.exp(-snll)
        ax[i+2].set_title(str(K) + ' Clusters    Prob: ' +
                          '{0:.2f}'.format(prob), fontsize=fontsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[i+2].spines[axis].set_linewidth(0.8)

    if save_name:
        plt.savefig(save_name, bbox_inches='tight')

    return K, np.exp(-sorted_nll[0])


def plot_avgs(losses, accs, rot_vars, w, save_name=None):
    """Plot training curve 
    """
    up = -1  # 3500

    avg_loss = []
    for i in range(w, len(losses)):
        avg_loss.append(np.mean(losses[i-w:i]))

    avg_acc = []
    for i in range(w, len(accs)):
        avg_acc.append(np.mean(accs[i-w:i]))

    avg_var = []
    for i in range(w, len(rot_vars)):
        avg_var.append(np.mean(rot_vars[i-w:i]))

    plt.figure(22, figsize=(13, 10))
    plt.clf()

    plt.subplot(312)
    plt.semilogy(avg_loss[:up])
    plt.ylabel('Mean NLL')
    plt.grid()

    plt.subplot(311)
    plt.plot(avg_acc[:up])
    plt.ylabel('Mean Accuracy')
    plt.grid()

    plt.subplot(313)
    plt.semilogy(avg_var[:up])
    plt.ylabel('NLL std/mean')
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)
        plt.close()


def combine_imgs(img_path_list, save_path, downsample=None):
    """Combine multiple images on disk
    """
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
    """Combine multiple images on disk 
    """
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



