
# train NCP
python main_ncp_spike_fast.py 

# if stopped in the middle, resume training from checkpoint 
python main_ncp_spike_fast.py saved_models/partially_trained_checkpoint.pt

# generate data
python infer_49ch_non_triaged_generate_data.py --N 2000 --do_corner_padding

# run NCP inference 
python infer_49ch_non_triaged_ncp.py inference_49ch_no_triage_N-2000_pad --S 150 --beam


# plot NCP inference results 
python infer_49ch_non_triaged_plot.py inference_49ch_no_triage_N-2000_pad/cluster_S-150-beam_it-18600 \\
    --min_cls_size 50 --n_seeds 1 --plot_type overlay

# plot tsne
# get NCP encoder output 
python infer_49ch_non_triaged_encoder_only.py inference_49ch_no_triage_N-2000_pad

# plot tsne of raw features and tsne of encoder outputs 
python infer_49ch_non_triaged_plot.py inference_49ch_no_triage_N-2000_pad/cluster_S-150-beam_it-18600 \\
    --min_cls_size 50 --n_seeds 1 --plot_type tsne
