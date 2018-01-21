import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
#
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#dataset
housing_data = datasets.load_boston()

#通过shuffle函数把数据顺序打乱
X,y = shuffle(housing_data.data,housing_data.target,random_state=7)
num_training = int(0.8*len(X))
X_train,y_train = X[:num_training],y[:num_training]
X_test,y_test = X[num_training:],y[num_training:]

#构建模型并训练
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,y_train)

ad_regrssor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ad_regrssor.fit(X_train,y_train)
#
# y_pred_dt = dt_regressor.predict(X_test)
# mse = mean_squared_error(y_test,y_pred_dt)
# print(round(mse,2))
#
# y_pred_ad = ad_regrssor.predict(X_test)
# mse = mean_squared_error(y_test,y_pred_ad)
# print(round(mse,2))

def plot_feature_importance(feature_importances,title,feature_name):
    feature_importances = 100.0*(feature_importances/max(feature_importances))
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0])+0.5
    plt.figure()
    plt.bar(pos,feature_importances[index_sorted],align='center')
    plt.xticks(pos,feature_importances[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()
print(dt_regressor.feature_importances_)
print(housing_data.feature_names)
plot_feature_importance(dt_regressor.feature_importances_,'Decision Tree regrssor',housing_data.feature_names)