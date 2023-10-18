import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
path = "F:\\A_MyWork\\大三课程相关\\数据挖掘\\02-线性回归分析\\02-回归\\data\\regress_data1.csv"
data = pd.read_csv(path)
# print(data.head())    

# 可视化
# 绘制散点图
# data.plot(kind='scatter', x='人口', y='收益', figsize=(12,8))
# # 设置x轴标签
# plt.xlabel('人口', fontsize=18)
# # 设置y轴标签，并将标签旋转为水平方向
# plt.ylabel('收益', rotation=0, fontsize=18)
# # 显示图形
# plt.show()

def computeCost(X, y, w):
    """
    计算线性回归模型的代价函数。    
    参数：
    X -- 特征矩阵，形状为 (n_samples, n_features)
    y -- 标签向量，形状为 (n_samples,1)
    w -- 权重向量，形状为 (n_features,1)
    
    返回值：
    代价函数的值
    """
    inner = np.power(X @ w - y, 2)  # 计算预测值与实际值之差的平方和
    return np.sum(inner) / (2 * len(X))  # 对平方和进行求和并除以样本数量的两倍，得到代价函数的值


#使用线性方程组计算得到线性回归模型的参数
def linear_regression_LSM(X, y):
   # 计算X的转置乘以X
   XTX = np.dot(X.T, X)
   
   # 计算XTX的逆矩阵
   XTX_inv = np.linalg.inv(XTX)
   
   # 将XTX的逆矩阵乘以X的转置乘以y，得到参数向量w
   w = np.dot(XTX_inv, np.dot(X.T, y))
   
   return w

#将一列名为'Ones'的值全为1的列插入到data的第一列位置。以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)
# print(data.head())

#做一些数据初始化
cols = data.shape[1]  # 获取data的列数
X = data.iloc[:,:cols-1]  # 获取除最后一列外的所有列作为特征矩阵X（ones,人口）  
y = data.iloc[:,cols-1:]  # 获取最后一列作为目标变量y  (收益)
# print(X.head())
# print(y.head())

# 转化X，y，初始化w
X = X.values
y = y.values
w = np.zeros((X.shape[1], 1))
# print(X.shape, w.shape, y.shape)

# 计算初始代价函数 j0
j0 = computeCost(X, y, w)
# print(j)

# 计算LSM模型参数
w = linear_regression_LSM(X,y,)
print(w)

# 计算拟合的训练模型的代价函数
j1 = computeCost(X, y, w) 
print(j1) 
