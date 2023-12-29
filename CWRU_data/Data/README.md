收集正常軸承、單點驅動端和風扇端缺陷的數據。驅動端軸承實驗的資料收集速度為 12,000 個樣本/秒和 48,000 個樣本/秒。所有風扇端軸承資料均以 12,000 個樣本/秒的速度收集。

資料檔採用 Matlab 格式。每個文件包含風扇和驅動端振動數據以及馬達轉速。對於所有文件，變數名稱中的以下項目表示：

DE－驅動端加速度計數據
FE——風扇端加速度計數據
BA - 基礎加速度計數據
time - 時間序列數據
RPM - 測試期間的轉速

normal bearing without fault (N)
bearing with single point fault at the inner raceway (IR)
bearing with single point fault at the outer raceway (OR)
bearing with single point fault at the ball (B)

軸承共三種故障，內圈、外圈、滾珠
0.007, 0.014, 0.021代表故障直徑7, 14, 21密耳
外圈故障有3, 6, 9點鐘方向故障，損傷點為載荷位置
每種故障下負載又分為0, 1, 2, 3馬力

Normal包含243938个数据，Ball包含121265个数据，InnerRace包含121991个数据， OuterRace包含122571个数据

猜測正常資料採樣頻率48kHZ

#Normal_0.mat; dict_keys(['X097_DE_time', 'X097_FE_time', 'X097RPM'])
#Normal_1.mat; dict_keys(['X098_DE_time', 'X098_FE_time'])
#Normal_2.mat; dict_keys(['ans', 'X098_DE_time', 'X098_FE_time', 'X099_DE_time', 'X099_FE_time'])
#Normal_3.mat; dict_keys(['X100_DE_time', 'X100_FE_time', 'X100RPM'])

#12k_DE
#B007_0.mat dict_keys(['X118_DE_time', 'X118_FE_time', 'X118_BA_time', 'X118RPM'])
#B014_0.mat dict_keys(['X185_DE_time', 'X185_FE_time', 'X185_BA_time', 'X185RPM'])
#B021_0.mat dict_keys(['X222_DE_time', 'X222_FE_time', 'X222_BA_time', 'X222RPM'])
#B028_0.mat dict_keys(['X048_DE_time'])

#IR007_0.mat dict_keys(['X105_DE_time', 'X105_FE_time', 'X105_BA_time', 'X105RPM'])
#OR007@3_0.mat dict_keys(['X144_DE_time', 'X144_FE_time', 'X144_BA_time', 'X144RPM'])