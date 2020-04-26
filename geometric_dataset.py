import torch
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm
import numpy as np

def edge_index_func(matrix_path):
    # print("In edge index function")
    a, b = [], []
    matrix = np.load(matrix_path)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # if(matrix[i][j] == 1 and i != j):
            if(matrix[i][j] == 1):
                a.append(i)
                b.append(j)
    edge = [a,b]
    edge_index = torch.tensor(edge, dtype=torch.long)
    return edge_index

class geometric_dataset(Dataset):
    def __init__(self, dataset_path, adjacency_matrix_path, dtw_matrix_path, node_num = 524, speed_mean = 29.0982979559, speed_std = 9.75304346669, his_num = 12, pred_num = 6, split_point_start = 0, split_point_end = 100*144, type = "Train"):

        # load data
        print("Geometric Dataset init start.")
        dataset = np.load(dataset_path)

        self.dataset = dataset[split_point_start:split_point_end]
        print(f"INFO: {type} dataset shape is", self.dataset.shape)

        self.edge_index = edge_index_func(adjacency_matrix_path)
        self.dtw_edge_index = edge_index_func(dtw_matrix_path)

        print("Geometric Dataset init finish.")

        self.node_num = node_num
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.his_num = his_num
        self.pred_num = pred_num

    def __getitem__(self, index):
        
        x_data = self.dataset[:,:-1][index: index + self.his_num]
        
        y_data = self.dataset[:,:-1][index + self.his_num: index + self.his_num + self.pred_num]
        y_data = y_data * self.speed_std + self.speed_mean

        x_i = torch.transpose(torch.tensor(x_data, dtype = torch.float), 1, 0)
        y_i = torch.transpose(torch.tensor(y_data, dtype = torch.float), 1, 0)
        edge_index_i = self.edge_index
        data_i = Data(x = x_i, edge_index = edge_index_i, y = y_i)
        data_i.dtw_edge_index = self.dtw_edge_index

        return data_i

    def __len__(self):
        data_length = self.dataset.shape[0] - self.pred_num - self.his_num
        return data_length

