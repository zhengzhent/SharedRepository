import pandas as pd
import numpy as np

x_data = pd.read_csv('''data/X_data.csv''').to_numpy()
y_label = pd.read_csv('data/y_label.csv').to_numpy()
theta1 = pd.read_csv('data/Theta1.csv', header=None).to_numpy()
theta2 = pd.read_csv('data/Theta2.csv', header=None).to_numpy()

# x_data.shape, y_label.shape, theta1.shape, theta2.shape

# 防止出现 overflow
def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(x - m)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
    
'''
输入通过在末尾添加一列全为1的列来增广，以考虑偏置项。
计算输入的增广形式与权重的转置的点积。
将结果通过 softmax 函数，得到层的最终输出
'''
def feedforward(weights, x):    
    augmented_input = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return softmax(np.dot(augmented_input, weights.T))


'''
对于每个数据点，将真实标签 y[i] 与相应预测中最大值的索引（np.argmax(pred[i]) + 1）进行比较。
'''

def score(x, y, theta1, theta2):
    x_hidden_data = feedforward(theta1, x)
    pred = feedforward(theta2, x_hidden_data)
    s = 0.0
    for i in range(pred.shape[0]):
        if y[i] == np.argmax(pred[i]) + 1:
            s += 1
    return s / pred.shape[0]

print("准确率为：",score(x_data, y_label, theta1, theta2))