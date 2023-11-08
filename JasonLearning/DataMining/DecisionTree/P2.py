from sklearn.model_selection import GridSearchCV, train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# import graphviz
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = "F:\\A_MyWork\\大三课程相关\\数据挖掘\\06 DecisionTree\\german_clean.csv"
data = pd.read_csv(path, header=1)
column = data.shape[1]
# print(column)
X = data.iloc[:,0:column-1]
y = data.iloc[:,column-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10) #分割训练集和测试集
# print(X_train)
# print(y_train)
param = {
    'criterion': ['gini'],
    'max_depth': [30, 50, 60, 100],
    'min_samples_leaf': [2, 3, 5, 10],
    'min_impurity_decrease': [0.1, 0.2, 0.5]
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=6)
grid.fit(X_train, y_train)
print('最优分类器:', grid.best_params_, '最优分数:', grid.best_score_)  # 得到最优的参数和分值

# clf = DecisionTreeClassifier(criterion = 'entropy')
clf = DecisionTreeClassifier(criterion = 'gini',max_depth=30,min_impurity_decrease=0.1,min_samples_leaf=2)
clf.fit(X_train, y_train,)
print(clf.score(X_test, y_test))
tree.plot_tree(clf)
plt.show()
