import pandas as pd
import numpy as np

# 读取数据
data1 = pd.read_excel('data1.xls')
data2 = pd.read_excel('data2.xls')

# 假设 X 和 Y 分别是 data1 和 data2 的某一列
X = data1['column_name']  # 替换为实际列名
Y = data2['column_name']  # 替换为实际列名

# 计算熵的函数
def entropy(data):
    probabilities = data.value_counts(normalize=True)
    return -np.sum(probabilities * np.log(probabilities))

# 计算 H(X), H(Y)
H_X = entropy(X)
H_Y = entropy(Y)

# 计算 H(X, Y)
joint_data = pd.concat([X, Y], axis=1)
H_XY = entropy(joint_data)

# 计算条件熵 H(X|Y) 和 H(Y|X)
def conditional_entropy(X, Y):
    joint_prob = pd.crosstab(Y, X, normalize='index')
    return -np.sum(joint_prob * np.log(joint_prob + 1e-10), axis=1).mean()

H_X_given_Y = conditional_entropy(X, Y)
H_Y_given_X = conditional_entropy(Y, X)

# 输出结果
print(f"H(X): {H_X}")
print(f"H(Y): {H_Y}")
print(f"H(X, Y): {H_XY}")
print(f"H(X|Y): {H_X_given_Y}")
print(f"H(Y|X): {H_Y_given_X}")