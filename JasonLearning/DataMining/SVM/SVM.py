import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import seaborn as sb

path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\07 支持向量机\\telco.csv"
#数据使用中位数进行填充
data = pd.read_csv(path)
for field in ['logtoll', 'logequi', 'logcard', 'logwire']:
    median = data[field].median()
    data[field].fillna(median, inplace=True)

#划分训练集和测试集
col = data.shape[1]
# print(col)
X = data.iloc[:,0:col-1]
y = data.iloc[:,col-1]
X  = StandardScaler().fit_transform(X)   #标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10) #分割训练集和测试集
# print(X_test.shape)
# print(y_test.shape)

'''
训练一个SVM分类器，进行 10 折交叉验证
'''
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test, y_test))
# 十折交叉验证
# print(cross_val_score(svm, X_train, y_train, scoring='f1', cv=10))

'''
调参，使用网格搜索GridSearch
'''
from sklearn.model_selection import GridSearchCV
 
#把要调整的参数以及其候选值 列出来；
param_grid = {"gamma":[0.001,0.01,0.1,1,10,100],
             "C":[0.001,0.01,0.1,1,10,100],
             }
print("Parameters:{}".format(param_grid))
grid_search = GridSearchCV(SVC(),param_grid,cv=5) #实例化一个GridSearchCV类
grid_search.fit(X_train,y_train) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
print("Test set score:{:.2f}".format(grid_search.score(X_test,y_test)))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.2f}".format(grid_search.best_score_))