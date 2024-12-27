import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

#pip install --upgrade scikit-learn threadpoolctl  解决了threadpoolctl冲突的问题？

# import os
# os.environ["OMP_NUM_THREADS"] = "1"  # 限制 OpenMP 使用的线程数
# os.environ["MKL_NUM_THREADS"] = "1"  # 限制 MKL 使用的线程数


# ----------------------------- 数据加载与预处理 ----------------------------- #
# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values - 1  # 将类别标签转为 0, 1, 2

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------- Nonlinear Autoencoder ----------------------------- #
class NonLinearAutoencoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.encoder_hidden_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.encoder_hidden_bias = np.zeros((1, hidden_dim))
        self.encoder_output_weights = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.encoder_output_bias = np.zeros((1, latent_dim))
        self.decoder_hidden_weights = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.decoder_hidden_bias = np.zeros((1, hidden_dim))
        self.decoder_output_weights = np.random.randn(hidden_dim, input_dim) * 0.01
        self.decoder_output_bias = np.zeros((1, input_dim))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def encode(self, X):
        hidden = self.relu(np.dot(X, self.encoder_hidden_weights) + self.encoder_hidden_bias)
        latent = np.dot(hidden, self.encoder_output_weights) + self.encoder_output_bias
        return hidden, latent
    
    def decode(self, latent):
        hidden = self.relu(np.dot(latent, self.decoder_hidden_weights) + self.decoder_hidden_bias)
        output = np.dot(hidden, self.decoder_output_weights) + self.decoder_output_bias
        return hidden, output
    
    def forward(self, X):
        encoder_hidden, encoded = self.encode(X)
        decoder_hidden, decoded = self.decode(encoded)
        return encoder_hidden, encoded, decoder_hidden, decoded

    def compute_loss(self, X, decoded):
        return np.mean((X - decoded) ** 2)

    def train(self, X, epochs=100, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            encoder_hidden, encoded, decoder_hidden, decoded = self.forward(X)
            loss = self.compute_loss(X, decoded)
            losses.append(loss)

            d_loss_decoded = -2 * (X - decoded) / X.shape[0]
            d_decoder_output_weights = np.dot(decoder_hidden.T, d_loss_decoded)
            d_decoder_output_bias = np.sum(d_loss_decoded, axis=0, keepdims=True)
            d_decoder_hidden = np.dot(d_loss_decoded, self.decoder_output_weights.T) * self.relu_derivative(decoder_hidden)
            d_decoder_hidden_weights = np.dot(encoded.T, d_decoder_hidden)
            d_decoder_hidden_bias = np.sum(d_decoder_hidden, axis=0, keepdims=True)

            d_loss_encoded = np.dot(d_decoder_hidden, self.decoder_hidden_weights.T)
            d_encoder_output_weights = np.dot(encoder_hidden.T, d_loss_encoded)
            d_encoder_output_bias = np.sum(d_loss_encoded, axis=0, keepdims=True)
            d_encoder_hidden = np.dot(d_loss_encoded, self.encoder_output_weights.T) * self.relu_derivative(encoder_hidden)
            d_encoder_hidden_weights = np.dot(X.T, d_encoder_hidden)
            d_encoder_hidden_bias = np.sum(d_encoder_hidden, axis=0, keepdims=True)

            self.encoder_hidden_weights -= learning_rate * d_encoder_hidden_weights
            self.encoder_hidden_bias -= learning_rate * d_encoder_hidden_bias
            self.encoder_output_weights -= learning_rate * d_encoder_output_weights
            self.encoder_output_bias -= learning_rate * d_encoder_output_bias
            self.decoder_hidden_weights -= learning_rate * d_decoder_hidden_weights
            self.decoder_hidden_bias -= learning_rate * d_decoder_hidden_bias
            self.decoder_output_weights -= learning_rate * d_decoder_output_weights
            self.decoder_output_bias -= learning_rate * d_decoder_output_bias

        return losses

# 训练 Nonlinear Autoencoder
input_dim = X_scaled.shape[1]
hidden_dim = 30
latent_dim = 2  # 降维到 2 维
autoencoder = NonLinearAutoencoder(input_dim, hidden_dim, latent_dim)
losses = autoencoder.train(X_scaled, epochs=100000, learning_rate=0.01)

# 获取 Autoencoder 降维后的特征
_, autoencoded_features, _, _ = autoencoder.forward(X_scaled)

# ----------------------------- Soft K-Means聚类 ----------------------------- #
class Soft_K_means:
    def fit(self, X, K=5, beta=3, max_iters=100):
        """
        :param X: 输入数据
        :param K: 聚类数
        :param beta: 控制软分配的参数
        :param max_iters: 最大迭代次数
        :return:
        """
        N = len(X)

        # just for testing
        fig = plt.figure(figsize=(20, 15))
        columns = 4
        rows = 5

        # initialize mean of clusters using K-Means++
        M = self.initialize_centroids(X, K)
        R = np.zeros(shape=(N, K))

        # calculate mean of clusters
        costs = []
        iterations = []
        converge = False
        iteration = -1

        while not converge and iteration < max_iters:
            iteration += 1
            iterations.append(len(iterations))

            # step 1: calculate cluster responsibilities
            for n in range(N):
                for k in range(K):
                    R[n][k] = np.exp(- beta * self.diff(X[n], M[k]))

            R /= R.sum(axis=1, keepdims=True)

            # step 2: recalculate mean of clusters
            for k in range(K):
                numerator = 0
                denominator = 0

                for n in range(N):
                    numerator += R[n, k] * X[n]
                    denominator += R[n, k]

                M[k] = numerator / denominator

            # compute cost
            costs.append(self.compute_cost(X, M, R))

            # check the convergence of the algorithm
            if len(costs) >= 2 and np.abs(costs[-1] - costs[-2]) < 1e-6:
                converge = True

            # just for testing
            if (iteration % 10 == 0 or converge):
                fig.add_subplot(rows, columns, int(np.ceil(iteration / 10)) + 1)
                plt.scatter(X[:, 0], X[:, 1])
                plt.scatter(M[:, 0], M[:, 1])
                plt.title('iteration ' + str(iteration))

        # just for testing
        plt.show()

        plt.plot(iterations, costs)
        plt.title('Cost of soft-kmeans over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

        return M, R

    def initialize_centroids(self, X, k):
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

    def diff(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def compute_cost(self, X, M, R):
        cost = 0
        for n in range(len(X)):
            for k in range(len(M)):
                cost += self.diff(X[n], M[k]) * R[n, k]
        return cost

# 设置聚类数 K
k = 3

s = Soft_K_means()
centroids, responsibilities = s.fit(autoencoded_feature, K=3, beta=10)


# 应用自定义 K-Means++ 算法
centroids, labels_autoencoder = kmeans_plus_plus(autoencoded_features, k)

# 评估聚类结果
silhouette_autoencoder = silhouette_score(autoencoded_features, labels_autoencoder)
ari_autoencoder = adjusted_rand_score(y, labels_autoencoder)
nmi_autoencoder = normalized_mutual_info_score(y, labels_autoencoder)

print("\nAutoencoder 降维后的聚类结果:")
print(f"Silhouette Score: {silhouette_autoencoder:.4f}")
print(f"Adjusted Rand Index (ARI): {ari_autoencoder:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_autoencoder:.4f}")

# ----------------------------- 可视化聚类结果 ----------------------------- #
plt.figure(figsize=(6, 6))
plt.scatter(autoencoded_features[:, 0], autoencoded_features[:, 1], c=labels_autoencoder, cmap='viridis', s=50, label='Cluster')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means++ on Autoencoder Features')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()