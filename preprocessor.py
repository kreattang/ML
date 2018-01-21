import numpy as np
filename = 'data_singlevar.txt'
X = []
y = []
with open(filename,'r') as f:
    for line in f.readlines():
        xt,yt = [float(i)for i in line.split(',')]
        X.append(xt)
        y.append(yt)
#print 原始数据
# print(X)
# print(y)
# print(len(X))
num_training = int(0.8*len(X))
num_test = len(X)-num_training
#把X的前80%作为训练集
X_train = np.array(X[:num_training]).reshape((num_training,1))
# print(X_train)
y_train = np.array(y[:num_training])
# print(y_train)
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])
# print(X_test)
# print(y_test)
#train model
from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train,y_train)

#
# import matplotlib.pyplot as plt
# # plt.figure(1)
# plt.scatter(X,y,color='green')
# plt.title('Raw data')
# plt.show()

# plt.figure()
# plt.scatter(X_train,y_train,color='blue')
# plt.scatter(X_test,y_test,color='red')
# plt.title('Bule:Traing Data   Red:Test Data')
# plt.show()
# import matplotlib.pyplot as plt
y_train_pred = linear_regressor.predict(X_train)
# plt.figure(1)
# plt.scatter(X_train,y_train,color='green')
# plt.plot(X_train,y_train_pred,color='black',linewidth=4)
# plt.title('A sample of Linear Model(This picture is about prediction in training data)')
# plt.show()
#
y_test_pred = linear_regressor.predict(X_test)
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,y_test,color='black',linewidth=4)
plt.title('A sample of Linear Model(This picture is about prediction in test data)')
plt.show()

import sklearn.metrics as sm
print("Linear model mean squared error in train:",round(sm.mean_squared_error(y_train,y_train_pred),2))
print("Linear model mean squared error in test :",round(sm.mean_squared_error(y_test,y_test_pred),2))

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train,y_train)
