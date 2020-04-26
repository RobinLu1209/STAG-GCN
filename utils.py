import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import math
import os

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)
    
    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
    return result