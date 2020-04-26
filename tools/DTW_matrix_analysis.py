'''
distance matrix: 用中心经纬度计算道路节点与道路节点之间的欧式距离
dtw distance matrix：用道路一周平均数据作为特征，计算道路节点与道路节点的dtw距离

w_{ij} = exp({-d}/{sigmoid})
'''

import numpy as np

# distance_matrix = np.load("distance_matrix.npy")
dtw_matrix = np.load("dtw_distance_matrix.npy")
std = np.std(dtw_matrix)
dtw_matrix = dtw_matrix / std
dtw_matrix = np.exp(-1 * dtw_matrix)

# print(dtw_matrix[0])

dtw_threshold = 0.83

count_min, count_max = 524, 0
count_zero = 0
count_avg = 0
min_index, max_index = 0, 0

matrix = np.identity(524)

for i in range(524):
    dtw_count_i = 0
    for j in range(524):
        if dtw_matrix[i][j] > dtw_threshold:
            dtw_count_i += 1
            matrix[i][j] = 1
    # print(f"i = {i}: DTW counts: {dtw_count_i}")
    count_avg += dtw_count_i
    if dtw_count_i == 1:
        count_zero += 1
    if dtw_count_i > count_max:
        count_max = dtw_count_i
        max_index = i
    if dtw_count_i < count_min:
        count_min = dtw_count_i
        min_index = i
print(f"Max: {count_max}, index = {max_index} \r\n Min: {count_min}, index = {min_index}")
print(f"Avg: {count_avg/524}")
print(f"Zero: {count_zero}")

np.save("dtw_dataset_matrix.npy", matrix)