import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
filename = 'data_singlevar.txt'

#输入空间
X = []

#输出空间
y = []

#读取文件
with open(filename,'r') as f:
    for line in f.readlines():
        xt,yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)


num_trainnig = int(0.8*len(X))
num_test = len(X)-num_trainnig

#分割数据集，80%训练集，20%测试集
#训练集
X_train = np.array(X[:num_trainnig]).reshape(num_trainnig,1)
y_train = np.array(y[:num_trainnig])

#测试集
X_test = np.array(X[num_trainnig:]).reshape(num_test,1)
y_test = np.array(y[num_trainnig:])

#导入sklearn
from sklearn import linear_model

#生成线性回归模型
linear_regressor = linear_model.LinearRegression()

#训练模型
linear_regressor.fit(X_train,y_train)



def linear_reg_figure(X_data,y_data,prde,text):
    plt.figure()
    plt.scatter(X_data, y_data, color='green')
    plt.plot(X_data, prde, color='black', linewidth=4)
    plt.title(text)
    plt.show()

#在训练集上的情况
y_train_pred = linear_regressor.predict(X_train)
# linear_reg_figure(X_train,y_train,y_train_pred,'linear Regressor')

#在测试集上的情况
y_test_pred = linear_regressor.predict(X_test)
# linear_reg_figure(X_test,y_test,y_test_pred,'test set')

#计算准确率
print("均方误差:",sm.mean_squared_error(y_test_pred,y_test))
print("平均绝对误差：",sm.mean_absolute_error(y_test,y_test_pred))
print("中位数绝对误差：",sm.median_absolute_error(y_test_pred,y_test))
print("解释方差：",sm.explained_variance_score(y_test,y_test_pred))
print("R方得分：",sm.r2_score(y_test_pred,y_test))






