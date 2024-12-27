# from lec11_svm_code.lec11_svm.rff import NormalRFF
from lec11_svm_code.lec11_svm.svc import BiLinearSVC, BiKernelSVC

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 移除标签为 2 的数据
data = data[data['class_label'] != 2]

# 4. 分离特征和目标值
features = data.iloc[:, :-1].values  # 除最后一列外的特征数据
target = data.iloc[:, -1].values  # 最后一列为目标值

# 5. 归一化
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

# 将目标值转换为二分类问题（将标签 1 保持为 1，将标签 3 转换为 0）
binary_target = np.where(target == 1, 1, 0)

# 打印 binary_target 的前 10 行
print(binary_target[:10])

# 6. 数据划分
X_train, X_test, y_train, y_test = train_test_split(normalized_features, binary_target, test_size=0.3, random_state=42)

# ----------------------------- 训练 BiLinearSVC 模型 ----------------------------- #
linear_svc = BiLinearSVC(C=1.0, max_iter=1000, tol=1e-5)
linear_svc.fit(X_train, y_train)
y_pred_linear = linear_svc.predict(X_test)

# 评估 BiLinearSVC 模型
accuracy_linear, precision_linear, recall_linear, f1_linear = accuracy_score(y_test, y_pred_linear), precision_score(y_test, y_pred_linear), recall_score(y_test, y_pred_linear), f1_score(y_test, y_pred_linear)
print("BiLinearSVC Model Evaluation:")
print(f"Accuracy: {accuracy_linear:.4f}")
print(f"Precision: {precision_linear:.4f}")
print(f"Recall: {recall_linear:.4f}")
print(f"F1 Score: {f1_linear:.4f}")

# ----------------------------- 训练 BiKernelSVC 模型 ----------------------------- #
kernel_svc = BiKernelSVC(C=1.0, kernel='rbf', max_iter=1000, tol=1e-5)
kernel_svc.fit(X_train, y_train)
y_pred_kernel = kernel_svc.predict(X_test)

# 评估 BiKernelSVC 模型
accuracy_kernel, precision_kernel, recall_kernel, f1_kernel = accuracy_score(y_test, y_pred_kernel), precision_score(y_test, y_pred_kernel), recall_score(y_test, y_pred_kernel), f1_score(y_test, y_pred_kernel)
print("\nBiKernelSVC Model Evaluation:")
print(f"Accuracy: {accuracy_kernel:.4f}")
print(f"Precision: {precision_kernel:.4f}")
print(f"Recall: {recall_kernel:.4f}")
print(f"F1 Score: {f1_kernel:.4f}")

# 设置绘图范围
x_min = np.min(X_train[:, 0])
x_max = np.max(X_train[:, 0])
y_min = np.min(X_train[:, 1])
y_max = np.max(X_train[:, 1])

plot_x = np.linspace(x_min - 1, x_max + 1, 1001)
plot_y = np.linspace(y_min - 1, y_max + 1, 1001)
xx, yy = np.meshgrid(plot_x, plot_y)

# 可视化决策边界（仅适用于二维特征）
def plot_decision_boundary(model, title, subplot_index):
    # 创建一个与训练数据相同维度的数组，前两个特征为网格点，其他特征为零
    X_full = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
    X_full[:, 0] = xx.ravel()
    X_full[:, 1] = yy.ravel()
    
    Z = model.decision_function(X_full)
    Z = Z.reshape(xx.shape)

    plt.subplot(2, 2, subplot_index)
    plt.contourf(xx, yy, Z, cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)

# 可视化决策边界（仅适用于二维特征）
def plot_decision_boundary(model, title, subplot_index):
    # 创建一个与训练数据相同维度的数组，前两个特征为网格点，其他特征为零
    X_full = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
    X_full[:, 0] = xx.ravel()
    X_full[:, 1] = yy.ravel()
    
    Z = model.decision_function(X_full)
    Z = Z.reshape(xx.shape)

    plt.subplot(2, 2, subplot_index)
    plt.contourf(xx, yy, Z, cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)

cmap = plt.cm.coolwarm

plt.figure(figsize=(10, 10))

# 可视化 BiLinearSVC 决策函数
plot_decision_boundary(linear_svc, 'SVC with linear kernel (Decision function view)', 1)

# 可视化 BiLinearSVC 预测结果
Z_linear_pred = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z_linear_pred = Z_linear_pred.reshape(xx.shape)
plt.subplot(2, 2, 2)
plt.contourf(xx, yy, Z_linear_pred, cmap=cmap)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel (0-1 view)')

# 可视化 BiKernelSVC 决策函数
plot_decision_boundary(kernel_svc, 'SVC with Gaussian kernel (Decision function view)', 3)

# 可视化 BiKernelSVC 预测结果
Z_kernel_pred = kernel_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z_kernel_pred = Z_kernel_pred.reshape(xx.shape)
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z_kernel_pred, cmap=cmap)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with Gaussian kernel (0-1 view)')

plt.show()