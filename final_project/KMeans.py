import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 分离特征和目标值
features = data.iloc[:, :-1]  # 除最后一列外的特征数据
target = data.iloc[:, -1]    # 最后一列为目标值

# 4. 归一化 (Normalization) 用于基于距离的模型（如 KNN、K-Means）
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

def initialize_centroids(X, k):
    """
    使用 K-Means++ 初始化质心
    """
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    # 随机选择第一个质心
    centroids[0] = X[np.random.randint(0, n_samples)]
    
    # 选择剩余的质心
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

def kmeans_plus_plus(X, k, max_iters=1000):
    """
    K-Means++ 算法
    """
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        # 分配每个样本到最近的质心
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
        
        # 计算新的质心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 检查质心是否收敛
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids, labels

# 设置聚类数 K
k = 3

# 应用 K-Means++ 算法
centroids, labels = kmeans_plus_plus(normalized_features, k)

# 打印质心
print("质心:\n", centroids)

# 计算轮廓系数
silhouette_avg = silhouette_score(normalized_features, labels)
print("轮廓系数 (Silhouette Score):", silhouette_avg)

# 计算调整兰德指数（需要真实标签）
ari = adjusted_rand_score(target, labels)
print("调整兰德指数 (Adjusted Rand Index):", ari)

# 计算互信息
nmi = normalized_mutual_info_score(target, labels)
print("互信息 (Normalized Mutual Information):", nmi)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(normalized_features[labels == i, 0], normalized_features[labels == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='*', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means++ Clustering')
plt.legend()
plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
# import matplotlib.pyplot as plt

# # 1. 读取数据
# data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# # 2. 设置列名
# columns = [
#     "area", "perimeter", "compactness", "length_of_kernel",
#     "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
# ]
# data.columns = columns

# # 3. 分离特征和目标值
# features = data.iloc[:, :-1]  # 除最后一列外的特征数据
# target = data.iloc[:, -1]    # 最后一列为目标值

# # 4. 归一化 (Normalization) 用于基于距离的模型（如 KNN、K-Means）
# normalizer = MinMaxScaler()
# normalized_features = normalizer.fit_transform(features)

# def initialize_centroids(X, k):
#     """
#     使用 K-Means++ 初始化质心
#     """
#     n_samples, n_features = X.shape
#     centroids = np.zeros((k, n_features))
    
#     # 随机选择第一个质心
#     centroids[0] = X[np.random.randint(0, n_samples)]
    
#     # 选择剩余的质心
#     for i in range(1, k):
#         distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
#         probabilities = distances / np.sum(distances)
#         cumulative_probabilities = np.cumsum(probabilities)
#         r = np.random.rand()
#         for j, p in enumerate(cumulative_probabilities):
#             if r < p:
#                 centroids[i] = X[j]
#                 break
                
#     return centroids

# def kmeans_plus_plus(X, k, max_iters=100):
#     """
#     K-Means++ 算法
#     """
#     centroids = initialize_centroids(X, k)
#     for _ in range(max_iters):
#         # 分配每个样本到最近的质心
#         distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
#         labels = np.argmin(distances, axis=0)
        
#         # 计算新的质心
#         new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
#         # 检查质心是否收敛
#         if np.all(centroids == new_centroids):
#             break
        
#         centroids = new_centroids
        
#     return centroids, labels

# # 设置聚类数 K
# k = 3

# # 应用 K-Means++ 算法
# centroids, labels = kmeans_plus_plus(normalized_features, k)

# # 打印质心
# print("质心:\n", centroids)

# # 计算轮廓系数
# silhouette_avg = silhouette_score(normalized_features, labels)
# print("轮廓系数 (Silhouette Score):", silhouette_avg)

# # 计算调整兰德指数（需要真实标签）
# ari = adjusted_rand_score(target, labels)
# print("调整兰德指数 (Adjusted Rand Index):", ari)

# # 计算互信息
# nmi = normalized_mutual_info_score(target, labels)
# print("互信息 (Normalized Mutual Information):", nmi)

# # 可视化聚类结果
# plt.figure(figsize=(10, 6))
# colors = ['r', 'g', 'b']
# for i in range(k):
#     plt.scatter(normalized_features[labels == i, 0], normalized_features[labels == i, 1], 
#                 s=50, c=colors[i], label=f'Cluster {i+1}')
# plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='*', label='Centroids')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-Means++ Clustering')
# plt.legend()
# plt.show()