# ncp-sort
Spike sorting using the [Neural Clustering Process (NCP)](https://github.com/aripakman/neural_clustering_process) algorithm. 

Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, Liam Paninski, [Discrete Neural Processes](https://arxiv.org/abs/1901.00409), arXiv:1901.00409

Set up: 
```
git clone https://github.com/yueqiw/ncp-sort.git
# python 3.6
pip install -r ncp-sort/requirements.txt
```
Store the training and testing data in `ncp-sort/data`.

Train NCP model using synthetic data: 
```
cd ncp-sort
python -m ncpsort.train_ncp --n_iter 10000 
(add --saved_checkpoint saved_models/{checkpoint}.pt for resuming from partially trained model
```

Run inference (probablistic clustering) on synthetic data:
```
# generate synthetic dataset
python -m ncpsort.cluster_synthetic_data.generate_synthetic_data \
        --output_dir inference_synthetic --N 1000 --n_seeds 10

# run clustering
python -m ncpsort.cluster_synthetic_data.inference_ncp_synthetic \
        --input_dir inference_synthetic_N-1000 \
        --model_file ./saved_models/NCP_10000.pt \
        --S 150 --beam --topn 2
```

Run inference (probablistic clustering) on real data:
```
#subset spikes from large recording data
python -m ncpsort.cluster_real_data.generate_data \
        --output_dir inference_real_data \
        --N 1000 --n_seeds 1 --do_corner_padding
        
# run clustering
python -m ncpsort.cluster_real_data.inference_ncp \
        --input_dir inference_real_data_N-1000_pad \
        --model_file ./saved_models/NCP_10000.pt \
        --S 150 --beam --topn 2
```

<br/>

<p align="center"> 
<img src="assets/fig1.png">
</p>

<p align="center"> 
Left: A detection module isolates multi-channel spike waveforms. The isolated waveforms are clustered by NCP. Right: Multiple sample cluster configurations from the NCP posterior, each indicating a visually plausible clustering of the data. 
</p>

<br/>

<p align="center"> 
<img src="assets/fig2.png">
</p>

<p align="center"> 
2000 spike waveforms from real data are clustered by NCP compared to vGMFM (variational inference on Gaussian Mixture of Finite Mixtures). 
</p>

<br/>


```
@misc{1901.00409,
Author = {Ari Pakman and Yueqi Wang and Catalin Mitelut and JinHyung Lee and Liam Paninski},
Title = {Discrete Neural Processes},
Year = {2019},
Eprint = {arXiv:1901.00409},
}
```
