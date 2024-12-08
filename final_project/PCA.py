import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

# PCA 实现
def pca(X, n_components=2):
    covariance_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 按特征值从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 选择前 n 个主成分
    principal_components = eigenvectors[:, :n_components]
    
    # 降维
    X_reduced = np.dot(X, principal_components)
    
    return X_reduced, principal_components

# 1. PCA 降维
X_reduced, principal_components = pca(X_scaled, n_components=2)

# 2. 数据重构
X_reconstructed = np.dot(X_reduced, principal_components.T)

# 3. 计算重构误差
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")

print("Original Data Shape:", X_scaled.shape)  # 原始数据
print("Reduced Data Shape:", X_reduced.shape)  # 降维后的数据
print("Principal Components Shape:", principal_components.shape)  # 主成分矩阵
print("Principal Components Transposed Shape:", principal_components.T.shape)  # 主成分转置
print("Reconstructed Data Shape:", X_reconstructed.shape)  # 重构后的数据



# 创建子图布局
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ["purple", "green", "yellow"]  # 定义三种类别对应的颜色

cmap = ListedColormap(colors)
# **子图 1: 编码器输出（离散颜色映射）**
sc = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap, edgecolor="k")
axes[0].set_title("Encoded Representation (Discrete Colors)")
axes[0].set_xlabel("Encoded Dimension 1")
axes[0].set_ylabel("Encoded Dimension 2")

# # 添加颜色条
# cb = fig.colorbar(sc, ax=axes[0], ticks=[1, 2, 3])
# cb.set_label("Class")
# cb.ax.set_yticklabels(['Class 1', 'Class 2', 'Class 3'])

# **子图 2: 原始数据与重构数据对比**
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], label="Original Data", alpha=0.5, c="blue")
axes[1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label="Reconstructed Data", alpha=0.5, c="red")
axes[1].legend()
axes[1].set_title("Original vs Reconstructed Data")
axes[1].set_xlabel("Feature 1 (Standardized)")
axes[1].set_ylabel("Feature 2 (Standardized)")

# 在子图中标注重构误差
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
axes[1].text(0.05, 0.95, f"Reconstruction Error: {reconstruction_error:.4f}",
             transform=axes[1].transAxes, fontsize=12, color="black", verticalalignment='top')

# 添加整体标题
fig.suptitle("PCA", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整子图布局，给主标题留出空间
plt.show()
