import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np
import math

import yaml
import torch
import argparse
import random
from geometric_dataset import geometric_dataset
from torch_geometric.data import Data, Dataset, DataLoader
from utils import *
from Models import STAG_GCN

def main(args):

    try:

        with open(args.config_filename) as f:
            config = yaml.load(f)
        
        data_args = config['data']
        model_args = config['model']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("INFO: device = ", device)
        
        model = STAG_GCN(node_num=data_args['node_num'], seq_len=model_args['his_num'], pred_len=model_args['pred_num'], graph_dim=model_args['graph_dim'], tcn_dim=model_args['tcn_dim'], atten_head=model_args['atten_head'], choice=model_args['choice']).to(device)
        # print(model)
        print(f"Model params: graph_dim = {model_args['graph_dim']}, tcn_dim={model_args['tcn_dim']}, atten_head = {model_args['atten_head']}")
        print('INFO: Model parameters_count:',count_parameters(model))
        model.load_state_dict(torch.load(model_args['model_filename']))
        print('INFO: Load model successful.')

        test_Dataset = geometric_dataset(dataset_path = data_args['dataset_path'],\
                                        adjacency_matrix_path = data_args['adjacency_matrix_path'],\
                                        dtw_matrix_path = data_args['dtw_matrix_path'],\
                                        node_num = data_args['node_num'],\
                                        speed_mean = data_args['speed_mean'],\
                                        speed_std = data_args['speed_std'],\
                                        his_num = model_args['his_num'], pred_num = model_args['pred_num'],\
                                        split_point_start = int(data_args['length'] * 0.8 * 144), split_point_end= int(data_args['length'] * 144), type='Test')
        test_dataloader = DataLoader(test_Dataset, batch_size = data_args['batch_size'], num_workers=8, pin_memory=True)

        print("INFO: Dataloader finish.")
        epochs = model_args['epochs']
        result_record = {}
        result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE'] = np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100]), np.array([100,100,100])

        model.eval()
        start = time.time()
        with torch.no_grad():
            for step_test, data in enumerate(test_dataloader):
                edge_index, dtw_edge_index = data.edge_index.to(device), data.dtw_edge_index.to(device)
                x_data = data.x.to(device)
                y_data = data.y.to(device)
                # print(f"x_data shape is {x_data.shape} y_data shape is {y_data.shape}")

                predictions = model(x_data, edge_index, dtw_edge_index)
                predictions, ground_truth = torch.reshape(predictions, (-1, data_args['node_num'], model_args['pred_num'])), torch.reshape(data.y, (-1, data_args['node_num'], model_args['pred_num']))

                pred_ = predictions.permute(0, 2, 1)
                y_ = ground_truth.permute(0, 2, 1)

                if step_test == 0:
                    prediction_result = pred_
                    ground_truth_result = y_
                else:
                    prediction_result = torch.cat((prediction_result, pred_), dim = 0)
                    ground_truth_result = torch.cat((ground_truth_result, y_), dim = 0)
            
        end = time.time()
        print(f"Testing time: {end - start}")

        prediction_result = prediction_result.cpu().numpy()
        ground_truth_result = ground_truth_result.cpu().numpy()

        result = metric_func(prediction_result, ground_truth_result, times=6)
        total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

        print("========== Evaluate results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
        print("---------------------------------------")
        
    except KeyboardInterrupt:
        MSE, RMSE, MAE, MAPE = result_record['MSE'], result_record['RMSE'], result_record['MAE'], result_record['MAPE']
        print("========== Evaluate results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print(f"[Config] name:{data_args['name']}, choice:{model_args['choice']}, graph_dim:{model_args['graph_dim']}")
        print("---------------------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
