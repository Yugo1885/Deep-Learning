'''ctr shift p 選取編譯器'''
'''***環境版本***
    env: tf2-gpu
    python -V: '3.7.16'
    tensorflow. __version__: '2.1.0'
    keras. __version__: '2.2.4-tf'
    nvcc -V: 'v10.1' #cuda version
    nvidia-smi #check gpu card info
    #cudnn version 7.6.1
    對應版本: "https://www.tensorflow.org/install/source_windows?hl=zh-tw#gpu"
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
from tensorflow_core.python.keras.layers import Flatten
from tensorflow_core.python.keras.optimizers import Adam
#from tensorflow_core.python.keras.models import Model
import tensorflow_core as tf

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

'''繪製曲線''' #比較訓練與測試差異
def plot_history(acc, val_acc, name:str):
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'r', label='Training '+name)
    plt.plot(epochs, val_acc, 'b', label='Validation '+name)
    plt.title('Training and validation '+name)
    plt.legend()
    #plt.figure()
    plt.show()
  
'''繪製混淆矩陣'''
def confusionMatrix(y_true:list, y_pred:list, labels=None):
    if labels is None:
        labels = [set(y_true+y_pred)]
    plt.title("Confusion Matrix")
    #plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_true, y_pred)
    #cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.matshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("True labels")
    plt.ylabel("Pred labels")
    plt.show()

'''繪製T-SNE'''
def t_sne():
    #xTSNE = mainfold.TSNE(n_components=2, init='pca', random_state=5, verbose=1).fit_transform(X)
    pass

'''驗證模型可靠性'''
#accuracy, precision, recall, f1-score
'''交叉驗證?'''

'''讀取資料路徑'''
def read_data(path:str):
    data = spio.loadmat(path)
    data = {k:v for k, v in data.items() if k[0] != '_'} #去除'_'開頭
    return data

#num_classes 狀態類別 有四種分類故num_classes = 4

'''資料前處理'''
def preprocess(data_list:list, num_samples=1000):
    #標準化?
    #歸一化: 將數據限制在小範圍，可以幫助快速收斂，提升模型性能
    #合併資料
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
#濾波器數量 filters
#卷積層大小 kernel_size
#卷積步長 strides
TIME_PERIODS = 1000 #The number of steps within one time segment
def build_model(inputShape=(TIME_PERIODS,), num_classes=4):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS,1), input_shape = inputShape))
    print(model)
    model.add(Conv1D(filters = 16, kernel_size = 3, strides = 1, activation='relu'))
    model.add(MaxPooling1D(2))  
    model.add(Conv1D(filters = 16, kernel_size = 3, strides = 1, activation='relu'))
    #model.add(GlobalAveragePooling1D()) #平均池化層，取兩個權重的平均值
    model.add(MaxPooling1D(2)) 
    model.add(Conv1D(filters = 32, kernel_size = 3, strides = 1, activation='relu'))
    model.add(MaxPooling1D(2)) 
    model.add(Flatten())
    model.add(Dropout(0.3)) #隨機選擇一半的神經元設為0，矩陣大小不變
    model.add(Dense(num_classes, activation='softmax')) #將長度為...的向量降為長度為num_classes的向量，因為有num_classes個類別要進行預測
    return(model)

'''MAIN'''
path_Normal = 'C:/Users/user/Desktop/DL/CWRU_data/Data/Normal/'
path_12kDE = 'C:/Users/user/Desktop/DL/CWRU_data/Data/12k_DE/'
Normal_0 = 'Normal_0.mat'
B007_0 = 'B007_0.mat'
IR007_0 = 'IR007_0.mat'
OR007_0 = 'OR007@3_0.mat'

NO_data = read_data(path_Normal+Normal_0)['X097_DE_time'].transpose()[0] #numpy.ndarray
B_data = read_data(path_12kDE+B007_0)['X118_DE_time'].transpose()[0]
IR_data = read_data(path_12kDE+IR007_0)['X105_DE_time'].transpose()[0]
OR_data = read_data(path_12kDE+OR007_0)['X144_DE_time'].transpose()[0]

all_data_list = [NO_data, B_data, IR_data, OR_data]

X_train, X_test, y_train, y_test = preprocess(all_data_list) 
model = build_model()
print(model.summary())
'''模型開始訓練'''
start_time = time.time()

#批次大小 batch_size設置越大 損失曲線來回震盪小
batch_size = 500
#訓練週期 
epochs = 100
#優化器
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
#optimizer = Adam(lr = 0.01)

'''編譯模型'''
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2) 
#validation_data=(X_test, y_test)
#ValueError: Error when checking input: expected reshape_input to have shape (100,) but got array with shape (1000,)
#長時間卡在"Successfully opened dynamic" 請確認cuda version

end_time = time.time()
total_time = end_time - start_time
print(f"模型訓練耗時: {total_time}s")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#print(history.history.keys())

plot_history(acc, val_acc, name='accuracy')
plot_history(loss, val_loss, name='loss')

'''
index = {0:'NO',1:'IR',2:'OR',3:'B'} #故障類別代號
y_true_labels = np.argmax(y_test, axis=0)
y_pred_labels = np.argmax(model.predict(X_test), axis=0)
confusionMatrix(y_true = y_true_labels, y_pred = y_pred_labels, labels=[index[i] for i in sorted(index)])
'''