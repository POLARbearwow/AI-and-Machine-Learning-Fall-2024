import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

# **2. 非线性自动编码器定义**
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
# **3. 初始化与训练模型**
input_dim = X_train.shape[1]  # 原始数据特征数
hidden_dim = 30  # 隐藏层神经元数量
latent_dim = 3  # 降维到 2 维
autoencoder = NonLinearAutoencoder(input_dim, hidden_dim, latent_dim)

# 训练模型并获取损失值
losses = autoencoder.train(X_train, epochs=100000, learning_rate=0.01)

# **4. 重构数据并计算重构误差**
_, encoded_data, _, X_reconstructed = autoencoder.forward(X_test)
reconstruction_error = autoencoder.compute_loss(X_test, X_reconstructed)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")

# **5. 可视化编码器的输出**
_, encoded_data, _, _ = autoencoder.forward(X_scaled)  # 获取编码器的低维表示

# **6. 使用离散颜色映射表绘图**
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图 1: 损失曲线
axes[0].plot(range(1, len(losses) + 1), losses, label="Training Loss")
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].grid(True)
axes[0].legend()

# 子图 2: 编码器输出（使用离散颜色映射）
colors = ["purple", "green", "yellow"]  # 定义三种类别对应的颜色
cmap = ListedColormap(colors)
sc = axes[1].scatter(encoded_data[:, 0], encoded_data[:, 1], c=y, cmap=cmap, edgecolor="k")
axes[1].set_title(f"Encoded Representation (2D) - Dimension: {latent_dim}")
axes[1].set_xlabel("Encoded Dimension 1")
axes[1].set_ylabel("Encoded Dimension 2")
cb = fig.colorbar(sc, ax=axes[1], ticks=[1, 2, 3])
cb.set_label("Class")
cb.ax.set_yticklabels(['Class 1', 'Class 2', 'Class 3'])

# 子图 3: 原始数据与重构数据对比
original_np = X_test
reconstructed_np = X_reconstructed
axes[2].scatter(original_np[:, 0], original_np[:, 1], label="Original Data", alpha=0.6, c="blue")
axes[2].scatter(reconstructed_np[:, 0], reconstructed_np[:, 1], label="Reconstructed Data", alpha=0.6, c="red")
axes[2].set_title("Original vs Reconstructed Data")
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
axes[2].legend()

# 在子图中标注重构误差
axes[2].text(0.05, 0.95, f"Reconstruction Error: {reconstruction_error:.4f}",
             transform=axes[2].transAxes, fontsize=12, color="black", verticalalignment='top')

# 添加整体标题
fig.suptitle("Nonlinear Autoencoder", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整子图布局，给主标题留出空间
plt.show()
