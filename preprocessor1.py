import numpy as np
from sklearn  import preprocessing

data = ([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
print("rew data:")
print(data)
data_standardized = preprocessing.scale(data)
#standardize the data
#对数据进行标准化

print("标准化")
print(data_standardized.mean(axis=0))
print(data_standardized.std(axis=0))

#均值缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
print("均值缩放")
print(data_scaler.fit_transform(data))

#归一化
print("归一化")
data_normalized = preprocessing.normalize(data,norm='l2')
print(data_normalized)


#二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print(data_binarized)

