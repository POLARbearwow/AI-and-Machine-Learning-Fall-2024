import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 数据加载函数
def load_data(train_path, test_path):
    # 加载训练和测试数据集
    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')
    
    # 分离特征（X）和标签（y）
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    return X_train, y_train, X_test, y_test

# 数据归一化函数
def normalize_data(X):
    # 将像素灰度值从 0-16 归一化到 0-1
    return X / 16.0

# 标签独热编码函数
def one_hot_encode(labels, num_classes=10):
    # 将标签转换为独热编码格式
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot

# 数据预处理
def preprocess_data(train_path, test_path):
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    # 归一化数据
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
    # 独热编码标签
    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test

# 数据集文件路径
train_file = './hw8/optdigits.tra'
test_file = './hw8/optdigits.tes'

# 调用数据预处理函数
X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test = preprocess_data(train_file, test_file)

# # 打印部分数据检查
# print("X_train shape:", X_train.shape)
# print("y_train_one_hot shape:", y_train_one_hot.shape)
# print("X_test shape:", X_test.shape)
# print("y_test_one_hot shape:", y_test_one_hot.shape)

class MLP:
    def __init__(self, units, activs):
        self.units = units
        self.length = len(units)
        self.activations = activs
        self.losses = []  # 用于记录每个 epoch 的平均损失
        assert len(units) - 1 == len(activs) and set(activs).issubset(
            set(["noactiv", "relu", "sigmoid", "softmax", "tanh"])
        ) and "softmax" not in activs[:-1]
        activDict, derivDict = MLP.Activations()
        self.activs = [None] + [activDict[i] for i in activs]
        self.derivs = [None] + [derivDict[i] if i != "softmax" else None for i in activs]
        self.Ws = [None] + [np.random.randn(units[i+1], units[i]) * 0.01 for i in range(len(units) - 1)]
        self.bs = [None] + [np.zeros((units[i+1], 1)) for i in range(len(units) - 1)]

    def Activations():
        noactiv = lambda x: x
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        relu = lambda x: np.maximum(0, x)
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        tanh = lambda x: np.tanh(x)
        noactiv_d = lambda x: np.ones_like(x)
        sigmoid_d = lambda x: sigmoid(x) * (1 - sigmoid(x))
        relu_d = lambda x: (x > 0).astype(float)
        tanh_d = lambda x: 1 - tanh(x)**2
        activations = {"noactiv": noactiv, "sigmoid": sigmoid, "relu": relu, "softmax": softmax, "tanh": tanh}
        derivatives = {"noactiv": noactiv_d, "sigmoid": sigmoid_d, "relu": relu_d, "tanh": tanh_d}
        return activations, derivatives

    def forward(self, X):
        Zs, As = [None] * self.length, [None] * self.length
        As[0] = X
        for i in range(1, self.length):
            Zs[i] = self.Ws[i] @ As[i-1] + self.bs[i]
            As[i] = self.activs[i](Zs[i])
        return Zs, As

    def _cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 避免 log(0) 问题
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    def backward(self, X, Y, lr):
        amount = X.shape[-1]
        Zs, As = self.forward(X)
        loss = self._cross_entropy_loss(Y, As[-1])
        dZ = As[-1] - Y
        for l in range(self.length - 1, 0, -1):
            dW = 1 / amount * dZ @ As[l-1].T
            db = 1 / amount * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA = self.Ws[l].T @ dZ
                dZ = dA * self.derivs[l-1](Zs[l-1])
            self.Ws[l] -= lr * dW
            self.bs[l] -= lr * db
        return loss

    def fit(self, X, Y, lr, max_iters, batch_size=None):
        assert X.shape[-1] == self.units[0]
        X, Y = X.T, Y.T
        amount = X.shape[-1]
        for epoch in range(max_iters):
            if not batch_size:
                loss_avg = self.backward(X, Y, lr)
            else:
                loss_avg = 0
                for i in range(math.ceil(amount / batch_size)):
                    loss = self.backward(
                        X[:, i*batch_size:(i+1)*batch_size],
                        Y[:, i*batch_size:(i+1)*batch_size],
                        lr
                    )
                    loss_avg = (i / (i+1)) * loss_avg + (1 / (i+1)) * loss
            self.losses.append(loss_avg)  # 保存每个 epoch 的损失值
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{max_iters}, Loss: {loss_avg:.4f}")

    def predict_proba(self, X_test):
        _, As = self.forward(X_test.T)
        return As[-1].T

    def predict(self, X_test):
        y_pred_prob = self.predict_proba(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred

# ------------ 模型初始化与训练 ------------

# 定义模型结构和激活函数
units = [64, 128, 64, 10]  # 输入层为 64, 隐藏层 128 和 64, 输出层为 10 类
activations = ["relu", "relu", "softmax"]

# 初始化 MLP 模型
mlp = MLP(units, activations)

# 训练模型
mlp.fit(
    X_train, y_train_one_hot,
    lr=0.01,
    max_iters=4000,
    batch_size=32
)

# 测试模型
y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ------------ 绘制损失函数 ------------

mlp_structure = "MLP Structure: Input Layer -> 128 Nodes (Hidden) -> 64 Nodes (Hidden) -> 10 Nodes (Output)"
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(mlp.losses) + 1), mlp.losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"MLP Loss Curve\n{mlp_structure}", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ------------ 计算评估指标 ------------

y_pred_proba = mlp.predict_proba(X_test)
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ------------ 绘制评估指标柱状图 ------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1.1)
plt.title(f"Evaluation Metrics (MLP)\n{mlp_structure}", fontsize=14)
plt.ylabel("Metric Value")
plt.xlabel("Metrics")

# 显示每个柱状图顶部的数值
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图像
plt.show()