import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

class Soft_K_means:
    def fit(self, X, K=3, beta=10, max_iters=100):
        """
        :param X: 输入数据
        :param K: 聚类数
        :param beta: 控制软分配的参数
        :param max_iters: 最大迭代次数
        :return:
        """
        N = len(X)
        M = self.initialize_centroids(X, K)  # 初始化质心
        R = np.zeros((N, K))  # 软分配矩阵
        costs = []  # 成本记录

        for iteration in range(max_iters):
            # Step 1: 计算簇责任分配
            for n in range(N):
                for k in range(K):
                    R[n, k] = np.exp(-beta * self.diff(X[n], M[k]))
            R /= R.sum(axis=1, keepdims=True)

            # Step 2: 更新质心
            for k in range(K):
                numerator = np.sum(R[:, k][:, np.newaxis] * X, axis=0)
                denominator = np.sum(R[:, k])
                M[k] = numerator / denominator

            # Step 3: 计算成本函数
            costs.append(self.compute_cost(X, M, R))

            # 检查收敛条件
            if len(costs) > 1 and np.abs(costs[-1] - costs[-2]) < 1e-6:
                break

        return M, R, costs

    def initialize_centroids(self, X, k):
        """
        使用 K-Means++ 初始化质心
        """
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

    def diff(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def compute_cost(self, X, M, R):
        cost = 0
        for n in range(len(X)):
            for k in range(len(M)):
                cost += self.diff(X[n], M[k]) * R[n, k]
        return cost


# ----------------------------- 数据加载与预处理 ----------------------------- #
if __name__ == '__main__':
    # 1. 读取数据
    data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

    # 2. 设置列名
    columns = [
        "area", "perimeter", "compactness", "length_of_kernel",
        "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
    ]
    data.columns = columns

    # 3. 分离特征和目标值
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1] - 1  # 类别标签转为 0, 1, 2

    # 4. 归一化
    normalizer = MinMaxScaler()
    normalized_features = normalizer.fit_transform(features)

    # ----------------------------- PCA 降维 ----------------------------- #
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(normalized_features)

    # ----------------------------- Kernel PCA 降维 ----------------------------- #
    kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    kernel_pca_features = kernel_pca.fit_transform(normalized_features)

    # ----------------------------- Soft K-Means 聚类 ----------------------------- #
    soft_kmeans = Soft_K_means()

    # PCA 降维后的 Soft K-Means
    centroids_pca, responsibilities_pca, costs_pca = soft_kmeans.fit(pca_features, K=3, beta=10)
    labels_pca = np.argmax(responsibilities_pca, axis=1)

    # Kernel PCA 降维后的 Soft K-Means
    centroids_kernel_pca, responsibilities_kernel_pca, costs_kernel_pca = soft_kmeans.fit(kernel_pca_features, K=3, beta=10)
    labels_kernel_pca = np.argmax(responsibilities_kernel_pca, axis=1)

    # ----------------------------- 子图可视化 ----------------------------- #
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 子图 1: PCA 聚类结果
    for i in range(3):
        axs[0, 0].scatter(pca_features[labels_pca == i, 0], pca_features[labels_pca == i, 1],
                          alpha=0.6, label=f'Cluster {i+1}')
    axs[0, 0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='yellow', s=300, marker='*', label='Centroids')
    axs[0, 0].set_title('Soft K-Means on PCA Features')
    axs[0, 0].legend()

    # 子图 2: Kernel PCA 聚类结果
    for i in range(3):
        axs[0, 1].scatter(kernel_pca_features[labels_kernel_pca == i, 0], kernel_pca_features[labels_kernel_pca == i, 1],
                          alpha=0.6, label=f'Cluster {i+1}')
    axs[0, 1].scatter(centroids_kernel_pca[:, 0], centroids_kernel_pca[:, 1], c='yellow', s=300, marker='*', label='Centroids')
    axs[0, 1].set_title('Soft K-Means on Kernel PCA Features')
    axs[0, 1].legend()

    # 子图 3: PCA 成本变化
    axs[1, 0].plot(range(len(costs_pca)), costs_pca, marker='o', label='Cost')
    axs[1, 0].set_title('Cost vs Iterations (PCA)')
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Cost')
    axs[1, 0].grid(True)

    # 子图 4: Kernel PCA 成本变化
    axs[1, 1].plot(range(len(costs_kernel_pca)), costs_kernel_pca, marker='o', label='Cost', color='orange')
    axs[1, 1].set_title('Cost vs Iterations (Kernel PCA)')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Cost')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------- 打印评估结果 ----------------------------- #
print("PCA 特征聚类结果:")
print(f"轮廓系数 (Silhouette Score): {silhouette_score(pca_features, labels_pca):.4f}")
print(f"调整兰德指数 (Adjusted Rand Index): {adjusted_rand_score(target, labels_pca):.4f}")
print(f"归一化互信息 (Normalized Mutual Information): {normalized_mutual_info_score(target, labels_pca):.4f}")

print("\nKernel PCA 特征聚类结果:")
print(f"轮廓系数 (Silhouette Score): {silhouette_score(kernel_pca_features, labels_kernel_pca):.4f}")
print(f"调整兰德指数 (Adjusted Rand Index): {adjusted_rand_score(target, labels_kernel_pca):.4f}")
print(f"归一化互信息 (Normalized Mutual Information): {normalized_mutual_info_score(target, labels_kernel_pca):.4f}")
