import numpy as np
import pandas as pd
import math
from math import log
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

path = "C:\\Users\\zheng_dpjpzv5\\Desktop\\实验6 决策树\\german_clean.csv"
data = pd.read_csv(path, header=0)
row,column = data.shape[0],data.shape[1]
X = data.iloc[:,0:column-1]
y = data.iloc[:,column-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10) #分割训练集和测试集
# print(X_train.shape)
# print(X_test.shape)
datasets = pd.concat([X_train,y_train],axis=1,verify_integrity = True)
# print(datasets) #构建训练集
labels = [u'checking_status', u'duration_new', u'credit_history', u'purpose', u'credit_amount',
          u'credit_amount_new', u'savings_status', u'employment', u'installment_commitment', u'personal_status',
          u'other_parties', u'residence_since', u'property_magnitude', u'age', u'other_payment_plans',
          u'housing', u'existing_credits', u'job', u'own_telephone', u'foreign_worker', u'class']


# 计算给定数据集的熵（信息熵）
def calc_ent(datasets):
    # 计算数据集的长度
    data_length = len(datasets)
    # 统计数据集中每个类别的出现次数
    label_count = {}
    for i in range(data_length):
        # 获取每个样本的标签
        label = datasets[i][-1]
        # 如果该类别不在label_count中，则添加到label_count中
        if label not in label_count:
            label_count[label] = 0
        # 统计该类别的出现次数
        label_count[label] += 1
    # 计算熵
    ent = -sum([(p / data_length) * log(p / data_length, 2)
                for p in label_count.values()])
    return ent

# 计算给定数据集在指定特征上的条件熵
def cond_ent(datasets, axis=0):
    # 计算数据集的长度
    data_length = len(datasets)
    # 使用字典feature_sets存储在指定特征上的不同取值对应的样本集合
    feature_sets = {}
    for i in range(data_length):
        # 获取每个样本在指定特征上的取值
        feature = datasets[i][axis]
        # 如果该取值不在feature_sets中，则添加到feature_sets中
        if feature not in feature_sets:
            feature_sets[feature] = []
        # 将该样本添加到对应取值的样本集合中
        feature_sets[feature].append(datasets[i])
    # 计算条件熵
    cond_ent = sum([(len(p) / data_length) * calc_ent(p)
                    for p in feature_sets.values()])
    return cond_ent

# 计算信息增益比
def info_gainRate(ent, cond_ent):
    # 信息增益比等于信息增益/信息熵 = （信息熵 - 条件熵）/信息熵
    return (ent - cond_ent)/ent

# 使用信息增益选择最佳特征作为根节点特征进行决策树的训练
def info_gain_train(datasets):
    # 计算特征的数量
    count = len(datasets[0]) - 1
    # 计算整个数据集的熵
    ent = calc_ent(datasets)
    # 存储每个特征的信息增益
    best_feature = []
    for c in range(count):
        # 计算每个特征的条件熵
        c_info_gain = info_gainRate(ent, cond_ent(datasets, axis=c))
        # 将特征及其对应的信息增益存入best_feature列表中
        best_feature.append((c, c_info_gain))
        # 输出每个特征的信息增益
        print('特征({}) 的信息增益为： {:.3f}'.format(labels[c], c_info_gain))
    # 找到信息增益最大的特征
    best_ = max(best_feature, key=lambda x: x[-1])
    # 返回信息增益最大的特征作为根节点特征
    return '特征({})的信息熵最大，选择为根节点特征'.format(labels[best_[0]])

print(info_gain_train(np.array(datasets)))

# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label:': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p)
                        for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :
                                               -1], train_data.iloc[:,
                                                                    -1], train_data.columns[:
                                                                                            -1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(
                    ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(
                    ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(
            root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] ==
                                          f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)
datatest = X_test.iloc[0]
# print(datatest.values)
print(dt.predict(datatest.values))