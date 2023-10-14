import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
path = "F:\\A_MyWork\\大三课程相关\\数据挖掘\\02-线性回归分析\\02-回归\\data\\traindata.csv"
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

'''
Batch Gradient Decent（批量梯度下降）
'''
def batch_gradientDescent(X, y, w, alpha, count):
    """
    批量梯度下降算法实现。
    
    参数：
    X -- 特征矩阵，形状为 (n_samples, n_features)
    y -- 标签向量，形状为 (n_samples,1)
    w -- 权重向量，形状为 (n_features,1)
    alpha -- 学习率
    count -- 迭代次数
    
    返回值：
    w -- 更新后的权重向量
    costs -- 每次迭代的代价函数值列表
    """
    # 初始化代价函数值列表
    costs = []

    # 对每个样本进行迭代
    for i in range(count):
        # 根据公式更新权重向量
        w = w - (X.T @ (X @ w - y)) * alpha / len(X)

        # 计算当前代价函数值并添加到列表中
        cost = computeCost(X, y, w)
        costs.append(cost)

        # 每隔100次迭代输出一次当前代价函数值
        if i % 100 == 0:
            print("在第{}次迭代中，cost的值是：{}。".format(i, cost))

    # 返回最终的权重向量和代价函数值列表
    return w, costs

# 特征归一化
data = (data - data.mean()) / data.std()#标准化数据
# print(data.head())

#将一列名为'Ones'的值全为1的列插入到data的第一列位置。以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)
# print(data.head())

#做一些数据初始化
cols = data.shape[1]  # 获取data的列数
X = data.iloc[:,:cols-1]  # 获取除最后一列外的所有列作为特征矩阵X（Size:50*501）  
y = data.iloc[:,cols-1:]  # 获取最后一列作为目标变量y  (Y:50*1)
print(X.head())
print(y.head())

# 转化X，y，初始化w
X = X.values
y = y.values
w = np.zeros((X.shape[1], 1))
# print(X.shape, w.shape, y.shape)

# 计算初始代价函数 j0
j0 = computeCost(X, y, w)
# print(j0)

# 设置学习速率和迭代次数
alpha = 0.01
iters = 2000
w, cost = batch_gradientDescent(X, y, w, alpha, iters)#返回更新后的参数向量g和损失值数组cost。

# 计算拟合的训练模型的代价函数
j1 = computeCost(X, y, w) 
print(j1) 

# # 生成预测值
# x = np.linspace(data['人口'].min(), data['人口'].max(), 100)
# # 根据参数估计值生成预测值
# f = w[0, 0] + (w[1, 0] * x)
# # 创建图形和轴对象
# fig, ax = plt.subplots(figsize=(12, 8))
# # 绘制预测值曲线
# ax.plot(x, f, 'r', label='预测值')
# # 绘制训练数据散点图
# ax.scatter(data['人口'], data['收益'], label='训练数据')
# # 添加图例
# ax.legend(loc=2)
# # 设置x轴和y轴标签
# ax.set_xlabel('人口', fontsize=18)
# ax.set_ylabel('收益', rotation=0, fontsize=18)
# # 设置图标题
# ax.set_title('预测收益和人口规模', fontsize=18)
# # 显示图形
# plt.show()

# 每一百次损失变化
# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(12, 8))
# 绘制代价随迭代次数的变化曲线
ax.plot(np.arange(iters), cost, 'r')
# 设置x轴和y轴标签
ax.set_xlabel('迭代次数', fontsize=18)
ax.set_ylabel('代价', rotation=0, fontsize=18)
# 设置图标题
ax.set_title('误差和训练Epoch数', fontsize=18)
# 显示图形
plt.show()
