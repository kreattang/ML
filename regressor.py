from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
X = []
y = []
with open('data_singlevar.txt','r') as f:
    for line in f.readlines():
        xt,yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)
# print(X)
# print(y)
num_training = int(0.8*len(X))
num_test = len(X)-num_training
# print(num_training)
# print(num_test)
#80%的数据用于训练,20%的数据用于测试
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

#构建模型
linear_regressor = linear_model.LinearRegression()
#训练
linear_regressor.fit(X_train,y_train)

#在训练集上的表现
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.subplot(2,2,1)
plt.scatter(X_train,y_train,edgecolors='g')
plt.plot(X_train,y_train_pred,color='b',linewidth=4)
plt.title("Training data")
#在测试集上的表现
plt.subplot(2,2,2)
y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test,y_test,edgecolors='g')
plt.plot(X_test,y_test_pred,color='b',linewidth=4)
plt.title("Test data")
plt.show()
#计算回归准确性

import sklearn.metrics as sm
#round是保留2位数字
print(round(sm.mean_absolute_error(y_test,y_test_pred),2))
print(round(sm.mean_squared_error(y_test,y_test_pred),2))




