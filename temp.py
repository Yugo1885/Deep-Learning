'''暫存區'''
import numpy as np
import os
import scipy.io as spio
from sklearn.model_selection import train_test_split 

from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.layers import Dense, Conv1D, Dropout, Reshape
from tensorflow_core.python.keras.layers import MaxPooling1D, GlobalAveragePooling1D
from tensorflow_core.python.keras.layers import Flatten
from tensorflow_core.python.keras.optimizers import Adam
import tensorflow_core as tf

'''讀取資料路徑'''
def read_data(path):
    data = spio.loadmat(path)
    data = {k:v for k, v in data.items() if k[0] != '_'} #去除'_'開頭
    return data

'''資料前處理'''
def preprocess(data_list:list, num_samples=100):
    subset_data_list = []
    for data in data_list:
        subset_data = data[:num_samples]
        subset_data_list.append([subset_data])
    combined_data = np.concatenate((subset_data_list), axis=0) #多個陣列按照水平方式合併
    X = np.array(combined_data) #X: 特徵資料
    y = np.array([0, 1, 2, 3]) #y: 標籤
    #打亂資料
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    #拆分資料 訓練集:測試集 = 8:2
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=0)
    #print("資料前處理完成")
    return X_train, X_test, y_train, y_test
  
'''模型架構'''
TIME_PERIODS = 100
def build_model(inputShape=(TIME_PERIODS,), num_classes=4):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS,1), input_shape = inputShape))
    print(model)
    model.add(Conv1D(16, 8, strides = 2,activation='relu'))
    model.add(MaxPooling1D(2))  
    #model.add(Conv1D(160, 10, activation='relu'))
    #model.add(GlobalAveragePooling1D()) #平均池化層，取兩個權重的平均值
    model.add(Flatten())
    model.add(Dropout(0.5)) #隨機選擇一半的神經元設為0，矩陣大小不變
    model.add(Dense(num_classes, activation='softmax')) #將長度為...的向量降為長度為num_classes的向量，因為有num_classes個類別要進行預測
    return(model)


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

X_train, X_test, y_train, y_test = preprocess(all_data_list) 
model = build_model()
print(model.summary())

batch_size = 32
epochs = 10
optimizer = tf.keras.optimizers.Adam(lr=0.01)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))






