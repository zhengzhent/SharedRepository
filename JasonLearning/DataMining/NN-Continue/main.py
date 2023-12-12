import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_x = 'D:\\MyWork\\Program\\JasonLearning\\DataMining\\NN-Continue\\data\\X_data.csv'
path_y = 'D:\\MyWork\\Program\\JasonLearning\\DataMining\\NN-Continue\\data\\y_label.csv'
template_X = pd.read_csv(path_x).values
template_y = pd.read_csv(path_y).values

template_total = np.concatenate((template_X, template_y), axis=1)
np.random.shuffle(template_total)
template_X = template_total[:, :-1]
template_y = template_total[:, -1].reshape((-1, 1))
X = template_X[:4000, :]
y = template_y[:4000, :].astype(int)
test_X = template_X[4000:, :]
test_y = template_y[4000:, :].astype(int)


def compute_accuracy(X_val, y_val, W1, b1, W2, b2):
    z1_val = np.dot(X_val, W1) + b1
    a1_val = np.maximum(0, z1_val)
    z2_val = np.dot(a1_val, W2) + b2
    exp_scores_val = np.exp(z2_val)
    probs_val = exp_scores_val / np.sum(exp_scores_val, axis=1, keepdims=True)

    predictions_val = np.argmax(probs_val, axis=1)
    accuracy_val = np.mean(predictions_val == y_val)

    return accuracy_val


# 步骤1：初始化参数
input_size = X.shape[1]
hidden_size = 400
output_size = 10
learning_rate = 0.01

# 初始化权重和偏置
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
y = np.squeeze(y)
y = y - 1
# 步骤2-6：迭代训练
num_iterations = 2000
result = []

for i in range(num_iterations):
    # 步骤2：前向传播
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU 激活函数
    z2 = np.dot(a1, W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax 激活函数

    # 步骤3：计算交叉熵损失
    correct_logprobs = -np.log(probs[range(len(X)), y])
    loss = np.sum(correct_logprobs) / len(X)

    # 步骤4：反向传播
    # 计算梯度
    dscores = probs
    dscores[range(len(X)), y] -= 1
    dscores /= len(X)

    dW2 = np.dot(a1.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dz1 = np.dot(dscores, W2.T) * (z1 > 0)  # ReLU 的导数
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # 步骤5：更新参数
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # 打印损失
    if i % 100 == 0:
        print(f"Eporch {i}, Loss: {loss}")
        result.append(loss)
        accuracy = compute_accuracy(X, y, W1, b1, W2, b2)
        print(f"Accuracy: {accuracy}")

# 保存参数
np.savez('test_array.npz', W1=W1, b1=b1, W2=W2, b2=b2)
# 打印损失图像
plt.plot(range(20), result)
plt.show()

# 测试集精度验证
test_y = np.squeeze(test_y)
test_y = test_y - 1

data = np.load('test_array.npz')
W1 = data['W1']
W2 = data['W2']
b1 = data['b1']
b2 = data['b2']
accuracy = compute_accuracy(test_X, test_y, W1, b1, W2, b2)
print(f"Accuracy: {accuracy}")
