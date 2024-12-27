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
        # 初始化权重和偏置
        self.encoder_hidden_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.encoder_hidden_bias = np.zeros((1, hidden_dim))
        self.encoder_output_weights = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.encoder_output_bias = np.zeros((1, latent_dim))
        self.decoder_hidden_weights = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.decoder_hidden_bias = np.zeros((1, hidden_dim))
        self.decoder_output_weights = np.random.randn(hidden_dim, input_dim) * 0.01
        self.decoder_output_bias = np.zeros((1, input_dim))
    
    def relu(self, x):
        # ReLU 激活函数
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        # ReLU 激活函数的导数
        return (x > 0).astype(float)
    
    def encode(self, X):
        # 编码器
        hidden = self.relu(np.dot(X, self.encoder_hidden_weights) + self.encoder_hidden_bias)
        latent = np.dot(hidden, self.encoder_output_weights) + self.encoder_output_bias
        return hidden, latent
    
    def decode(self, latent):
        # 解码器
        hidden = self.relu(np.dot(latent, self.decoder_hidden_weights) + self.decoder_hidden_bias)
        output = np.dot(hidden, self.decoder_output_weights) + self.decoder_output_bias
        return hidden, output
    
    def forward(self, X):
        # 前向传播
        encoder_hidden, encoded = self.encode(X)
        decoder_hidden, decoded = self.decode(encoded)
        return encoder_hidden, encoded, decoder_hidden, decoded

    def compute_loss(self, X, decoded):
        # 均方误差 (MSE)
        return np.mean((X - decoded) ** 2)

    def train(self, X, epochs=100, learning_rate=0.01):
        # 训练过程
        losses = []
        for epoch in range(epochs):
            # 前向传播
            encoder_hidden, encoded, decoder_hidden, decoded = self.forward(X)
            loss = self.compute_loss(X, decoded)
            losses.append(loss)

            # 反向传播（计算梯度）
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

            # 梯度下降更新参数
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
losses = autoencoder.train(X_scaled, epochs=1000, learning_rate=0.01)

# 获取 Autoencoder 降维后的特征
_, autoencoded_features, _, _ = autoencoder.forward(X_scaled)

# ----------------------------- K-Means++ 聚类 ----------------------------- #
def apply_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)  # 添加 n_init=10
    labels = kmeans.fit_predict(X)
    return labels, kmeans

# Autoencoder 降维后的 K-Means++
labels_autoencoder, kmeans_autoencoder = apply_kmeans(autoencoded_features, n_clusters=3)

# ----------------------------- 聚类评估 ----------------------------- #
def evaluate_clustering(X, labels, y_true):
    silhouette = silhouette_score(X, labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return silhouette, ari, nmi

# 评估 Autoencoder 降维后的聚类
silhouette_autoencoder, ari_autoencoder, nmi_autoencoder = evaluate_clustering(autoencoded_features, labels_autoencoder, y)
print("\nAutoencoder 降维后的聚类结果:")
print(f"Silhouette Score: {silhouette_autoencoder:.4f}")
print(f"Adjusted Rand Index (ARI): {ari_autoencoder:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_autoencoder:.4f}")

# ----------------------------- 可视化聚类结果 ----------------------------- #
plt.figure(figsize=(6, 6))
plt.scatter(autoencoded_features[:, 0], autoencoded_features[:, 1], c=labels_autoencoder, cmap='viridis', s=50, label='Cluster')
plt.scatter(kmeans_autoencoder.cluster_centers_[:, 0], kmeans_autoencoder.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means++ on Autoencoder Features')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()
