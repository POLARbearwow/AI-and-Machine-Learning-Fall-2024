#注释部分不知为何无法使用
# from lec11_svm_code.lec11_svm.rff import NormalRFF
from lec11_svm_code.lec11_svm.svc import BiLinearSVC, BiKernelSVC

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 分离特征和目标值
features = data.iloc[:, :-1].values  # 除最后一列外的特征数据
target = data.iloc[:, -1].values  # 最后一列为目标值

# 4. 归一化
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

# 打印 target 的前 10 行
print(target[:10])

# 5. 数据划分
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# 使用 SMOTE 平衡数据集
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 使用带权重的 SVM 模型
linear_svc = OneVsRestClassifier(
    BiLinearSVC(C=1.0, max_iter=1000, tol=1e-5)
)
kernel_svc = OneVsRestClassifier(
    BiKernelSVC(C=1.0, kernel="rbf", max_iter=1000, tol=1e-5)
)

# 训练模型
linear_svc.fit(X_resampled, y_resampled)
kernel_svc.fit(X_resampled, y_resampled)

# 预测
y_pred_linear = linear_svc.predict(X_test)
y_pred_kernel = kernel_svc.predict(X_test)

# 评估 BiLinearSVC 模型性能
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear, average='weighted', zero_division=0)
recall_linear = recall_score(y_test, y_pred_linear, average='weighted', zero_division=0)
f1_linear = f1_score(y_test, y_pred_linear, average='weighted', zero_division=0)

print("BiLinearSVC Model Evaluation:")
print(f"Accuracy: {accuracy_linear:.4f}")
print(f"Precision: {precision_linear:.4f}")
print(f"Recall: {recall_linear:.4f}")
print(f"F1 Score: {f1_linear:.4f}")

# 评估 BiKernelSVC 模型性能
accuracy_kernel = accuracy_score(y_test, y_pred_kernel)
precision_kernel = precision_score(y_test, y_pred_kernel, average='weighted', zero_division=0)
recall_kernel = recall_score(y_test, y_pred_kernel, average='weighted', zero_division=0)
f1_kernel = f1_score(y_test, y_pred_kernel, average='weighted', zero_division=0)

print("\nBiKernelSVC Model Evaluation:")
print(f"Accuracy: {accuracy_kernel:.4f}")
print(f"Precision: {precision_kernel:.4f}")
print(f"Recall: {recall_kernel:.4f}")
print(f"F1 Score: {f1_kernel:.4f}")


# # ----------------------------- 训练 BiLinearSVC 模型 ----------------------------- #
# linear_svc = OneVsRestClassifier(BiLinearSVC(C=1.0, max_iter=1000, tol=1e-5))
# linear_svc = BiLinearSVC(C=1.0, max_iter=1000, tol=1e-5)
# linear_svc.fit(X_train, y_train)
# y_pred_linear = linear_svc.predict(X_test)

# # 评估 BiLinearSVC 模型
# accuracy_linear = accuracy_score(y_test, y_pred_linear)
# precision_linear = precision_score(y_test, y_pred_linear, average='weighted')
# recall_linear = recall_score(y_test, y_pred_linear, average='weighted')
# f1_linear = f1_score(y_test, y_pred_linear, average='weighted')
# print("BiLinearSVC Model Evaluation:")
# print(f"Accuracy: {accuracy_linear:.4f}")
# print(f"Precision: {precision_linear:.4f}")
# print(f"Recall: {recall_linear:.4f}")
# print(f"F1 Score: {f1_linear:.4f}")

# # ----------------------------- 训练 BiKernelSVC 模型 ----------------------------- #
# # kernel_svc = OneVsRestClassifier(BiKernelSVC(C=1.0, kernel='rbf', max_iter=1000, tol=1e-5))
# kernel_svc = BiKernelSVC(C=1.0, kernel='rbf', max_iter=1000, tol=1e-5)
# kernel_svc.fit(X_train, y_train)
# y_pred_kernel = kernel_svc.predict(X_test)

# # 评估 BiKernelSVC 模型
# accuracy_kernel = accuracy_score(y_test, y_pred_kernel)
# precision_kernel = precision_score(y_test, y_pred_kernel, average='weighted')
# recall_kernel = recall_score(y_test, y_pred_kernel, average='weighted')
# f1_kernel = f1_score(y_test, y_pred_kernel, average='weighted')
# print("\nBiKernelSVC Model Evaluation:")
# print(f"Accuracy: {accuracy_kernel:.4f}")
# print(f"Precision: {precision_kernel:.4f}")
# print(f"Recall: {recall_kernel:.4f}")
# print(f"F1 Score: {f1_kernel:.4f}")

# # 可视化决策边界（仅适用于二维特征）
# def plot_decision_boundary(X, y, model, title):
#     h = .02  # step size in the mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
    
#     # 创建一个与训练数据相同维度的数组，前两个特征为网格点，其他特征为零
#     X_full = np.zeros((xx.ravel().shape[0], X.shape[1]))
#     X_full[:, 0] = xx.ravel()
#     X_full[:, 1] = yy.ravel()
    
#     Z = model.predict(X_full)
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title(title)
#     plt.show()

# # 仅可视化前两个特征
# plot_decision_boundary(X_train, y_train, linear_svc, 'BiLinearSVC Decision Boundary')
# plot_decision_boundary(X_train, y_train, kernel_svc, 'BiKernelSVC Decision Boundary')