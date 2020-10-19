# STAG-GCN
Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting [\[paper\]](https://dl.acm.org/doi/10.1145/3340531.3411894)

![Spatiotemporal Adaptive Gated Graph Convolution Network](figures/system_model.png "Model Architecture")

## Requirements
- pytorch >= 1.4.0
- numpy >= 1.18.1
- scikit-learn >= 0.21.0
- pytorch geometric >= 1.4.1
- pyaml
- scipy
- tqdm

## Data 
The data in paper can be download here: [GAIA Open Dataset](https://outreach.didichuxing.com/research/opendata/)

## Graph Construction
Run the following command to generate semantic neighbor adjacency matrix.
```bash
# Achieve DTW distance matrix
python tools/DTW_embedding.py
# Set threshold to generate semantic neighbor adjacency matrix
python tools/DTW_matrix_analysis.py
```

## Model Training & Testing
```bash
# Training process
python train.py --config_filename='config.yaml'
# Testing process
python test.py --config_filename='config.yaml'
```
## Citation
If you find this repository, e.g., the paper, code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{lu2020staggcn,
author = {Lu, Bin and Gan, Xiaoying and Jin, Haiming and Fu, Luoyi and Zhang, Haisong},
title = {Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting},
year = {2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management},
pages = {1025–1034},
numpages = {10},
location = {Virtual Event, Ireland},
series = {CIKM '20}
}
```
