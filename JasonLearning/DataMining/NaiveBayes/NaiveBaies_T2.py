import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from collections import Counter
import math
import matplotlib.pyplot as plt

# 数据读入
path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验4 朴素贝叶斯网络\\weather_nominal_encode.csv"
data = pd.read_csv(path, header=None,skiprows=1)
# print(data.head)
cols = data.shape[1]
X = data.iloc[:,:cols-1]  
y = data.iloc[:,cols-1:]

# 随机划分训练集 测试集 3 ：7
# print(dataX.head)
# print(datay.head)
# rowX = dataX.shape[0]
# rowy = datay.shape[0]
# print(rowX,rowy)
# X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=0.3)
# print(X_train)
# print(X_test)

class NaiveBayes:
    def __init__(self):
        self.model = None
    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))
    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))
    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries
    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        """
        训练模型
        :param X: 特征数据
        :param y: 标签数据
        :return: 训练结果
        """
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'
    # 计算概率
    def calculate_probabilities(self, input_data):
        """
        计算概率
        :param input_data: 待预测的数据
        :return: 概率字典
        """
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities
    # 预测类别
    def predict(self, X_test):
        """
        预测类别
        :param X_test: 测试数据
        :return: 预测结果
        """
        label = sorted(self.calculate_probabilities(X_test).items(),
                       key=lambda x: x[-1])[-1][0]
        return label
    def score(self, X_test, y_test):
        """ 
        计算准确率
        :param X_test: 测试数据
        :param y_test: 测试标签
        :return: 准确率
        """
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))

clf = GaussianNB()
accuracy = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(clf.fit(X_train, y_train))
    temp = clf.score(X_test, y_test)
    accuracy.append(temp)
    # print(clf.score(X_test, y_test))
print(accuracy)
fig, ax = plt.subplots(figsize=(12, 8))
# 绘制代价随迭代次数的变化曲线
ax.plot(np.arange(100), accuracy, 'r')
# 设置x轴和y轴标签
ax.set_xlabel('num', fontsize=18)
ax.set_ylabel('accuuracy', rotation=0, fontsize=18)
# 设置图标题
ax.set_title('accuracy-num', fontsize=18)
# 显示图形
plt.show()