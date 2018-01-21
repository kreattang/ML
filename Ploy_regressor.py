import numpy as np
import sklearn.metrics as sm

filename = 'data_multivar.txt'
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        y.append(yt)
num_training = int(0.8*len(X))
num_test = len(X)-num_training
# print(num_training)
# print(num_test)
#80%的数据用于训练,20%的数据用于测试
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

from sklearn.preprocessing import PolynomialFeatures
ploynomial = PolynomialFeatures(degree=3)
X_train_transformed = ploynomial.transform(X_train)

data = [0.39,2.78,7.11]

from sklearn import linear_model
ploy_linear_model = linear_model.LinearRegression()
ploy_linear_model.fit(X_train_transformed,y_train)
print(ploy_linear_model.predict(ploynomial.transform((data))))

