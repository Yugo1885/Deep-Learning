'''CWRU資料集前處理測試區'''
import numpy as np
import os
import scipy.io as spio

'''讀取資料路徑'''
def read_data(path):
    data = spio.loadmat(path)
    data = {k:v for k, v in data.items() if k[0] != '_'} #去除'_'開頭
    return data

def preprocess(data_list: list, num_samples=100):
    subset_data_list = []
    for data in data_list:
        subset_data = data[:num_samples]
        subset_data_list.append(subset_data)
    combined_data = np.concatenate((subset_data_list), axis=0)
    return combined_data

path_Normal = 'CWRU_data/Data/Normal/'
path_12kDE = 'CWRU_data/Data/12k_DE/'
Normal_0 = 'Normal_0.mat'
B007_0 = 'B007_0.mat'
IR007_0 = 'IR007_0.mat'
OR007_0 = 'OR007@3_0.mat'

NO_data = read_data(path_Normal+Normal_0)['X097_DE_time'].transpose()[0]
B_data = read_data(path_12kDE+B007_0)['X118_DE_time'].transpose()[0]
IR_data = read_data(path_12kDE+IR007_0)['X105_DE_time'].transpose()[0]
OR_data = read_data(path_12kDE+OR007_0)['X144_DE_time'].transpose()[0]

all_data_list = [NO_data, B_data, IR_data, OR_data]

combined_data = preprocess(all_data_list)