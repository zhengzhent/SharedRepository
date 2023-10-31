import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

class KNN:

    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    # 对给定的输入数据进行预测
    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            # 计算输入数据和训练样本之间的距离
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 将距离和对应的训练样本标签添加到knn_list中
            knn_list.append((dist, self.y_train[i]))

        # 对于训练集中不在前n个最近邻中的样本，找到距离最近的前n个邻居中出现次数最多的类别
        for i in range(self.n, len(self.X_train)):
            # 找到距离最大的点的索引
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            # 计算输入数据和训练样本之间的距离
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 如果当前样本的距离比最大距离小，则替换最大距离的样本
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 统计每个类别的出现次数
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        # 找到出现次数最多的类别
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0  # 初始化正确预测的数量为0
        n = 10  # 设置每个样本最多考虑的最近邻居数量为10
        for X, y in zip(X_test, y_test):  # 对于测试集中的每一个样本和标签
            label = self.predict(X)  # 使用模型进行预测
            if label == y:  # 如果预测结果与真实标签相同
                right_count += 1  # 则正确预测的数量加1
        return right_count / len(X_test)  # 返回模型在测试集上的准确率

path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验5 KNN算法\\cancer_X.csv"
dataX = pd.read_csv(path, header=None)
path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验5 KNN算法\\cancer_y.csv"
datay = pd.read_csv(path,header=None)
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
clf = KNN(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)