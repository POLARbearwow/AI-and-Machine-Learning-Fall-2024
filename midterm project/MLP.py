import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, recall_score, precision_score, f1_score
import math

"----------------------------------data preprocessing-------------------------------------------------"
#region data preprocessing

# 数据预处理
data = pd.read_csv('./midterm project/ai4i2020.csv')
data['Type'] = data['Type'].astype('category').cat.codes
data = data.drop(['UDI', 'Product ID'], axis=1)

features_to_scale = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

X = data.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1).values  # 特征
y = data['Machine failure'].values  # 标签

X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加偏置项

def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

value_counts = pd.Series(y).value_counts()
N_pos = value_counts[1]  # 正样本数量
N_neg = value_counts[0]  # 负样本数量

class MLP:
    def __init__(self, units, activs, w1, w2):
        self.units = units
        self.length = len(units)
        self.activations = activs
        self.w1 = w1  # 正样本的权重
        self.w2 = w2  # 负样本的权重
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

    def _weighted_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(
            self.w1 * y_true * np.log(y_pred) + self.w2 * (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, X, Y, lr):
        amount = X.shape[-1]
        Zs, As = self.forward(X)
        loss = self._weighted_cross_entropy_loss(Y, As[-1])
        if self.activations[-1] == "softmax":
            dZ = As[-1] - Y
        else:
            dA = -(self.w1 * Y / As[-1] - self.w2 * (1 - Y) / (1 - As[-1]))
        for l in range(self.length - 1, 0, -1):
            if self.activations[-1] != "softmax" or l < self.length - 1:
                dZ = dA * self.derivs[l](Zs[l])
            dW = 1 / amount * dZ @ As[l-1].T
            db = 1 / amount * np.sum(dZ, axis=1, keepdims=True)
            dA = self.Ws[l].T @ dZ
            self.Ws[l] -= lr * dW
            self.bs[l] -= lr * db
        return loss

    def fit(self, X, Y, lr, max_iters, batch_size=None):
        assert X.shape[-1] == self.units[0]
        X, Y = X.T, Y.reshape(-1, 1).T
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
            print(f"Epoch {epoch+1}/{max_iters}, Loss: {loss_avg}")

    def predict_proba(self, X_test):
        _, As = self.forward(X_test.T)
        return As[-1].T

    def predict(self, X_test, threshold=0.5):
        y_pred_prob = self.predict_proba(X_test)
        y_pred = (y_pred_prob >= threshold).astype(int)
        return y_pred

# ------------ 模型初始化与训练 ------------
units = [X_train.shape[1], 16, 8, 1]
activations = ["sigmoid", "sigmoid", "sigmoid"]
mlp = MLP(units, activations, N_pos, N_neg)

mlp.fit(
    X_train, y_train,
    lr=0.00001,
    max_iters=100000,
    batch_size=32
)

# ------------ 绘制损失函数 ------------
mlp_structure = "MLP Structure: Input Layer -> 16 Nodes (Hidden) -> 8 Nodes (Hidden) -> 1 Node (Output)"


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(mlp.losses) + 1), mlp.losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"MLP loss\n{mlp_structure}", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 计算评估指标
y_pred_proba = mlp.predict_proba(X_test)
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# MLP 的层次结构
mlp_structure = "MLP Structure: Input Layer -> 16 Nodes (Hidden) -> 8 Nodes (Hidden) -> 1 Node (Output)"

# 准备数据
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]  # 确保使用计算结果变量，而不是函数名

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1.1)
plt.title(f"Loss Curve (MLP)\n{mlp_structure}", fontsize=14)
plt.ylabel("Metric Value")
plt.xlabel("Metrics")

# 显示每个柱状图顶部的数值
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图像
plt.show()