import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read Data
path = "F:\\A_MyWork\\大三课程相关\\数据挖掘\\03Logistic_Regression\\telco.csv"
data = pd.read_csv(path, header=None,skiprows=1)
# print(data.head())
# shape : 100 * 3

# 可视化
# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Exam 1'],
#            positive['Exam 2'],
#            s=50,
#            c='b',
#            marker='o',
#            label='Admitted')
# ax.scatter(negative['Exam 1'],
#            negative['Exam 2'],
#            s=50,
#            c='r',
#            marker='x',
#            label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

# define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#define costReg function 
def costReg(w, X, y, learningRate):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * w.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * w.T)))
    reg = (learningRate /
           (2 * len(X))) * np.sum(np.power(w[:, 1:w.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

#计算一个梯度的步长函数 
def gradientReg(w, X, y, learningRate):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(w.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * w.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + (
                (learningRate / len(X)) * w[:, i])

    return grad

def batch_gradientDescentReg(w,X, y,alpha, count,learningRate):
    costsList = []
    # 对每个样本进行迭代
    for i in range(count):
        # 根据公式更新权重向量
        w = w - alpha * gradientReg(w,X,y,learningRate)
        # 计算当前代价函数值并添加到列表中

        costs = costReg(w,X,y,learningRate)
        costsList.append(costs)
        # 每隔100次迭代输出一次当前代价函数值
        if i % 100 == 0:
            print("在第{}次迭代中，cost的值是：{}。".format(i, costs))

    # 返回最终的权重向量和代价函数值列表
    return w, costsList


# add a column x0 = 1 - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]# 获取特征
y = data.iloc[:, cols - 1:cols]# 获取标签

# convert to numpy arrays and initalize the parameter array w
X = np.array(X.values)
y = np.array(y.values)
w = np.zeros((X.shape[1]))  
#此处的w为一维数组，因为当计算cost的时候进行了转置，不同于线性回归中直接设置成了X.shape[1]*1的矩阵
# print(X.shape, w.shape, y.shape)

# 计算初始化代价函数（初始化w=0）
# print(cost(w,X,y))

# 计算初始w=0的梯度下降结果
# print(gradient(w,X,y))
learningRate = 1
alpha = 0.00000006
iters = 4000
w, costs = batch_gradientDescentReg(w,X,y,alpha, iters,learningRate)  #返回更新后的参数向量g和损失值数组costs。
print(costReg(w, X, y,learningRate))    #final_lose 2.00772

def predict(w, X):
    probability = sigmoid(X @ w)
    return [1 if x >= 0.5 else 0 for x in probability]
# 计算精度
predictions = predict(w, X)
correct = [
    1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
    for (a, b) in zip(predictions, y)
]
accuracy = (sum(map(int, correct)) / len(correct))
print('accuracy = {0}'.format(accuracy))
print('w:',w)
plt.figure()
plt.plot(range(4000), costs, label="loss")
plt.legend()
plt.grid()
plt.show()







