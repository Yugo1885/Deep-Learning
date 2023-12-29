'''ctr shift p 選取編譯器'''
'''***環境版本***
    env: tf2-gpu
    python -V: '3.7.16'
    tensorflow. __version__: '2.1.0'
    keras. __version__: '2.2.4-tf'
'''
'''此為四分類模型區分Normal, B, IR, OR在各種故障直徑、0HP故障負載下的情形'''
'''***所用檔案***
    Normal_0, 12k_DE下的
    B007_0, 
    IR007_0,
    OR007@3_0
'''

from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers import Dense, Conv1D, Dropout, Reshape
from tensorflow_core.python.keras.layers import MaxPooling1D, GlobalAveragePooling1D
'''
    Dense 全連接層
    Conv1D 一維卷積層
    MaxPooling1D
    Dropout 防止過度擬合
    GlobalAveragePooling1D 一維數據全局最大池化層
'''
import os
import numpy as np
import pandas as pd
import scipy.io as spio
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split 
import time

#start_time = time.time()

fault_label = {0:'NO',1:'IR',2:'OR',3:'B'} #故障類別

#label:x; data:y
#x_train, y_train, x_test, y_test =
#批次大小 batch_size
#訓練週期 epochs
#濾波器數量 filters
#卷積層大小 kernel_size
#卷積步長 strides

'''繪製混淆矩陣'''
def confusion_matrix():

    plt.title("Confusion Matrix")
    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

'''繪製T-SNE'''
def t_sne():
    xTSNE = mainfold.TSNE(n_components=2, init='pca', random_state=5, verbose=1).fit_transform(X)
    pass

'''讀取資料路徑'''
def read_data(path:str):
    data = spio.loadmat(path)
    data = {k:v for k, v in data.items() if k[0] != '_'} #去除'_'開頭
    return data

#num_classes 狀態類別
#有四種分類故num_classes = 4


#train data and test data split ratio 訓練集 測試集 驗證集比例 80% 10% 10%
train_ratio = 0.8
val_ratio = 0.1

'''資料前處理'''

def preprocess(data_list:list, num_samples=100):
    #標準化?
    #歸一化: 將數據限制在小範圍，可以幫助快速收斂，提升模型性能
   
    ##打亂資料
  

    #合併資料
    subset_data_list = []
    for data in data_list:
        subset_data = data[:num_samples]
        subset_data_list.append(subset_data)
    combined_data = np.concatenate((subset_data_list), axis=0)
    X = np.array(combined_data) #X: 特徵資料
    y = np.array([0, 1, 2, 3]) #y: 標籤
    #拆分資料 20%測試集 80%訓練集
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, y_train, X_test, y_test


'''模型建構'''
TIME_PERIODS = 6000
def build_model(input_shape=(TIME_PERIODS,), num_classes=10):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))

    model.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D()) #平均池化層，取兩個權重的平均值
    model.add(Dropout(0.5)) #隨機選擇一半的神經元設為0，矩陣大小不變
    model.add(Dense(num_classes, activation='softmax')) #將長度為...的向量降為長度為num_classes的向量，因為有num_classes個類別要進行預測
    return(model)
    #print(model.summary())

'''編譯模型'''
#compile model
'''驗證模型可靠性'''
#accuracy, precision, recall, f1-score
'''交叉驗證?'''

'''***main***'''
path_Normal = 'CWRU_data/Data/Normal/'
path_12kDE = 'CWRU_data/Data/12k_DE/'
Normal_0 = 'Normal_0.mat'
B007_0 = 'B007_0.mat'
IR007_0 = 'IR007_0.mat'
OR007_0 = 'OR007@3_0.mat'

NO_data = read_data(path_Normal+Normal_0)['X097_DE_time'].transpose()[0] #numpy.ndarray
B_data = read_data(path_12kDE+B007_0)['X118_DE_time'].transpose()[0]
IR_data = read_data(path_12kDE+IR007_0)['X105_DE_time'].transpose()[0]
OR_data = read_data(path_12kDE+OR007_0)['X144_DE_time'].transpose()[0]

all_data_list = [NO_data, B_data, IR_data, OR_data]

combined_data = preprocess(all_data_list) 

#取前?筆資料 打亂順序 合併
