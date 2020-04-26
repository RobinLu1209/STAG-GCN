'''
Use DTW algorithm to calculate the feature distance of two road
and each two sequence is 144-length-time-series. 
'''

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# from generate_matrix import load_matrix
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

pwd = '../data/'
path = 'dataset.npy'
matrix_path = 'matrix.npy'

node_num = 524

data = np.load(pwd + path)
df = pd.DataFrame(data)
df['symbol'] = df[node_num]%(7 * 24 * 6)

for i in tqdm(range(7 * 24 * 6)):
    df_i = df[df['symbol'] == i]
    values_i = df_i.values[:,:-2]
    mean_i = np.mean(values_i, axis = 0)[np.newaxis, :]
    if(i == 0):
        mean = mean_i
    else:
        mean = np.concatenate((mean, mean_i), axis=0)

mean = mean.T
# speed_mean = 29.0982979559
# speed_std = 9.75304346669
# mean = mean * speed_std + speed_mean
print("INFO: mean shape:", mean.shape)
matrix = np.zeros((node_num, node_num))

for index_x in tqdm(range(node_num)):
    for index_y in range(index_x, node_num):
        x = mean[index_x]
        y = mean[index_y]
        distance, _ = fastdtw(x, y, dist=euclidean)
        matrix[index_x][index_y] = distance
# f.close()

for i in range(524):
    for j in range(0,i):
        matrix[i][j] = matrix[j][i]

np.save("dtw_distance_matrix.npy", matrix)