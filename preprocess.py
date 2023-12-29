'''CWRU資料集前處理測試區'''
import numpy as np
import scipy.io as spio

#path
normal_path = 'CWRU_data/Data/Normal/'
de12k_path = 'CWRU_data/Data/12k_DE/OR007@3_0.mat'

normal_fileName = 'Normal_3.mat'
lib = spio.loadmat(normal_path+normal_fileName)
#Normal_0=scio.loadmat(Normal0_DIR)['X097_FE_time'].transpose()[0] #列向量轉行向量
lib2 = spio.loadmat(de12k_path)

def remove_dash(lib):
    lib = {k:v for k, v in lib.items() if k[0] != '_'} #去除'_'開頭
    return lib

#print(type(lib))
lib = remove_dash(lib)
lib2 = remove_dash(lib2)
print(normal_fileName,lib.keys())
print(de12k_path,lib2.keys())
#print(lib['ans'])
#print(len(lib2['X118_DE_time'])) #122571
#print(len(lib2['X118_FE_time'])) #122571

'''
for i in lib2.keys():
    print(i)
'''
#print(len(lib['X097_FE_time'])) #共243938
#print(list(lib["X097_FE_time"][:100]))


