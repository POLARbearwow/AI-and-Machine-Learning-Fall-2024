import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# ----------------------------- 数据加载与预处理 ----------------------------- #
# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 分离特征和目标
features = data.iloc[:, :-1]
target = data.iloc[:, -1] - 1  # 将类别标签转为 0, 1, 2

# 4. 特征归一化
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

# ----------------------------- PCA 降维 ----------------------------- #
# 使用 PCA 将特征降维到 2 维
pca = PCA(n_components=2)
pca_features = pca.fit_transform(normalized_features)

# ----------------------------- Kernel PCA 降维 ----------------------------- #
# 使用 Kernel PCA（非线性降维）将特征降维到 2 维
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
kernel_pca_features = kernel_pca.fit_transform(normalized_features)

# ----------------------------- K-Means++ 聚类 ----------------------------- #
def initialize_centroids(X, k):
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    centroids[0] = X[np.random.randint(0, n_samples)]
    for i in range(1, k):
        distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[i] = X[j]
                break
    return centroids

def kmeans_plus_plus(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    loss_history = []  # 用于记录每次迭代的损失
    for _ in range(max_iters):
        # 计算每个样本到每个质心的距离
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
        
        # 计算损失
        loss = np.sum([np.linalg.norm(X[labels == i] - centroids[i], axis=1).sum() for i in range(k)])
        loss_history.append(loss)

        # 更新质心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 如果质心没有变化，则停止迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels, loss_history

# 设置聚类数
k = 3

# PCA 降维后的 K-Means++
centroids_pca, labels_pca, loss_history_pca = kmeans_plus_plus(pca_features, k)

# Kernel PCA 降维后的 K-Means++
centroids_kernel_pca, labels_kernel_pca, loss_history_kernel_pca = kmeans_plus_plus(kernel_pca_features, k)

# ----------------------------- 可视化聚类结果 ----------------------------- #
plt.figure(figsize=(12, 6))

# 子图 1: PCA 降维的聚类结果
plt.subplot(1, 2, 1)
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(pca_features[labels_pca == i, 0], pca_features[labels_pca == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='yellow', marker='*', label='Centroids')
plt.title('K-Means++ on PCA Features')
plt.legend()

# 子图 2: Kernel PCA 降维的聚类结果
plt.subplot(1, 2, 2)
for i in range(k):
    plt.scatter(kernel_pca_features[labels_kernel_pca == i, 0], kernel_pca_features[labels_kernel_pca == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids_kernel_pca[:, 0], centroids_kernel_pca[:, 1], s=300, c='yellow', marker='*', label='Centroids')
plt.title('K-Means++ on Kernel PCA Features')
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------- 可视化损失曲线 ----------------------------- #
plt.figure(figsize=(12, 6))

# 子图 3: PCA 聚类损失变化
plt.subplot(1, 2, 1)
plt.plot(loss_history_pca, marker='o', color='blue', label='Loss')
plt.title('Loss vs Iterations (PCA)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 子图 4: Kernel PCA 聚类损失变化
plt.subplot(1, 2, 2)
plt.plot(loss_history_kernel_pca, marker='o', color='orange', label='Loss')
plt.title('Loss vs Iterations (Kernel PCA)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------- 打印评估结果 ----------------------------- #
print("PCA 降维后的聚类结果:")
print(f"轮廓系数 (Silhouette Score): {silhouette_score(pca_features, labels_pca):.4f}")
print(f"调整兰德指数 (Adjusted Rand Index): {adjusted_rand_score(target, labels_pca):.4f}")
print(f"归一化互信息 (Normalized Mutual Information): {normalized_mutual_info_score(target, labels_pca):.4f}")

print("\nKernel PCA 降维后的聚类结果:")
print(f"轮廓系数 (Silhouette Score): {silhouette_score(kernel_pca_features, labels_kernel_pca):.4f}")
print(f"调整兰德指数 (Adjusted Rand Index): {adjusted_rand_score(target, labels_kernel_pca):.4f}")
print(f"归一化互信息 (Normalized Mutual Information): {normalized_mutual_info_score(target, labels_kernel_pca):.4f}")
