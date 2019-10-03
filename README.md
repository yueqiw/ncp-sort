# Spike sorting using the Neural Clustering Process (NCP)

This repo implements a spike sorting algorithm based on the Neural Clustering Process (NCP) [1,2,3], a recently introduced neural architecture that performs efficient amortized approximate Bayesian inference for probabilistic clustering. 

[1] Yueqi Wang, Ari Pakman, Catalin Mitelut, JinHyung Lee, Liam Paninski, **Spike Sorting using the Neural Clustering Process**, NeurIPS 2019 [Neuro AI Workshop](https://sites.google.com/mila.quebec/neuroaiworkshop) -- Real Neurons & Hidden Units: Future directions at the intersection of neuroscience
and artificial intelligence.

[2] Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, Liam Paninski, **Discrete Neural Processes**, arXiv:1901.00409 
https://arxiv.org/abs/1901.00409

[3] The NCP algorithm: https://github.com/aripakman/neural_clustering_process

## Code
```
git clone https://github.com/yueqiw/ncp-sort.git
# python 3.6
pip install -r ncp-sort/requirements.txt
```
Store the training and testing data in `ncp-sort/data`.

### Train NCP model using synthetic data: 
```
cd ncp-sort
python -m ncpsort.train_ncp --n_iter 10000 
```

### Run probablistic clustering on synthetic data:
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

### Run probablistic clustering on real data:
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

## Cite

```
@misc{1901.00409,
Author = {Ari Pakman and Yueqi Wang and Catalin Mitelut and JinHyung Lee and Liam Paninski},
Title = {Discrete Neural Processes},
Year = {2019},
Eprint = {arXiv:1901.00409},
}
```
