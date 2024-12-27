# '''
# Implementation of soft-kmeans
# '''
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# class Soft_K_means:
#     def fit(self, X, K=5, beta=3, max_iters=100):
#         """
#         :param X: 输入数据
#         :param K: 聚类数
#         :param beta: 控制软分配的参数
#         :param max_iters: 最大迭代次数
#         :return:
#         """
#         N = len(X)

#         # just for testing
#         fig = plt.figure(figsize=(20, 15))
#         columns = 4
#         rows = 5

#         # initialize mean of clusters using K-Means++
#         M = self.initialize_centroids(X, K)
#         R = np.zeros(shape=(N, K))

#         # calculate mean of clusters
#         costs = []
#         iterations = []
#         converge = False
#         iteration = -1

#         while not converge and iteration < max_iters:
#             iteration += 1
#             iterations.append(len(iterations))

#             # step 1: calculate cluster responsibilities
#             for n in range(N):
#                 for k in range(K):
#                     R[n][k] = np.exp(- beta * self.diff(X[n], M[k]))

#             R /= R.sum(axis=1, keepdims=True)

#             # step 2: recalculate mean of clusters
#             for k in range(K):
#                 numerator = 0
#                 denominator = 0

#                 for n in range(N):
#                     numerator += R[n, k] * X[n]
#                     denominator += R[n, k]

#                 M[k] = numerator / denominator

#             # compute cost
#             costs.append(self.compute_cost(X, M, R))

#             # check the convergence of the algorithm
#             if len(costs) >= 2 and np.abs(costs[-1] - costs[-2]) < 1e-6:
#                 converge = True

#             # just for testing
#             if (iteration % 10 == 0 or converge):
#                 fig.add_subplot(rows, columns, int(np.ceil(iteration / 10)) + 1)
#                 plt.scatter(X[:, 0], X[:, 1])
#                 plt.scatter(M[:, 0], M[:, 1])
#                 plt.title('iteration ' + str(iteration))

#         # just for testing
#         plt.show()

#         plt.plot(iterations, costs)
#         plt.title('Cost of soft-kmeans over iterations')
#         plt.xlabel('Iteration')
#         plt.ylabel('Cost')
#         plt.show()

#         return M, R

#     def initialize_centroids(self, X, k):
#         """
#         使用 K-Means++ 初始化质心
#         """
#         n_samples, n_features = X.shape
#         centroids = np.zeros((k, n_features))
        
#         # 随机选择第一个质心
#         centroids[0] = X[np.random.randint(0, n_samples)]
        
#         # 选择剩余的质心
#         for i in range(1, k):
#             distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
#             probabilities = distances / np.sum(distances)
#             cumulative_probabilities = np.cumsum(probabilities)
#             r = np.random.rand()
#             for j, p in enumerate(cumulative_probabilities):
#                 if r < p:
#                     centroids[i] = X[j]
#                     break
                    
#         return centroids

#     def diff(self, x, y):
#         return np.sqrt(np.sum((x - y) ** 2))

#     def compute_cost(self, X, M, R):
#         cost = 0
#         for n in range(len(X)):
#             for k in range(len(M)):
#                 cost += self.diff(X[n], M[k]) * R[n, k]
#         return cost


# if __name__ == '__main__':
#     # 1. 读取数据
#     data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

#     # 2. 设置列名
#     columns = [
#         "area", "perimeter", "compactness", "length_of_kernel",
#         "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
#     ]
#     data.columns = columns

#     # 3. 分离特征和目标值
#     features = data.iloc[:, :-1]  # 除最后一列外的特征数据
#     target = data.iloc[:, -1]    # 最后一列为目标值

#     # 4. 归一化 (Normalization) 用于基于距离的模型（如 KNN、K-Means）
#     normalizer = MinMaxScaler()
#     normalized_features = normalizer.fit_transform(features)

#     # 应用 Soft K-Means 算法
#     s = Soft_K_means()
#     centroids, responsibilities = s.fit(normalized_features, K=3, beta=10)

#     # 获取每个数据点的簇标签
#     labels = np.argmax(responsibilities, axis=1)

#     # 计算轮廓系数
#     silhouette_avg = silhouette_score(normalized_features, labels)
#     print("轮廓系数 (Silhouette Score):", silhouette_avg)

#     # 计算调整兰德指数（需要真实标签）
#     ari = adjusted_rand_score(target, labels)
#     print("调整兰德指数 (Adjusted Rand Index):", ari)

#     # 计算互信息
#     nmi = normalized_mutual_info_score(target, labels)
#     print("互信息 (Normalized Mutual Information):", nmi)

#     # 可视化聚类结果
#     plt.figure(figsize=(10, 6))
#     colors = ['r', 'g', 'b']
#     for i in range(3):
#         plt.scatter(normalized_features[:, 0], normalized_features[:, 1], 
#                     c=responsibilities[:, i], cmap='viridis', label=f'Cluster {i+1}', alpha=0.5)
#     plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='*', label='Centroids')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Soft K-Means Clustering')
#     plt.legend()
#     plt.show()