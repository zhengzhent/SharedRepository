import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验5 KNN算法\\cancer_X.csv"
dataX = pd.read_csv(path, header=None)
path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验5 KNN算法\\cancer_y.csv"
datay = pd.read_csv(path,header=None)

X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=0.3)
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# trainNum = X_train.shape[0]
# testNum = X_test.shape[0]
# print(trainNum , testNum)

# k=3时的结果
# clf_sk = KNeighborsClassifier(n_neighbors=3)
# # 用X_train和y_train去拟合
# clf_sk.fit(X_train, y_train)
# score = clf_sk.score(X_test, y_test)
# print(score)

best_score = 0.0
best_k = -1
for k in range(3, 111):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score

print("best_k = " + str(best_k))
print("best_score = " + str(best_score))



