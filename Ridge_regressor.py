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

from sklearn import linear_model
ridge_regressor = linear_model.Ridge(alpha=0.01,fit_intercept=True)
ridge_regressor.fit(X_train,y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)

print("mean_absolute_error:",sm.mean_absolute_error(y_test,y_test_pred_ridge))
print("mean_squared_error:",sm.mean_squared_error(y_test,y_test_pred_ridge))
print("median_absolute_error",sm.median_absolute_error(y_test,y_test_pred_ridge))
print("r2_score:",sm.r2_score(y_test,y_test_pred_ridge))

