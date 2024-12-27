import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

# ----------------------------- 数据加载与处理 ----------------------------- #
# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 将 class_label 转换为 0, 1, 2 的格式
data = data[data['class_label'] != 2]
data['class_label'] = data['class_label'] - 1  # 转换为从 0 开始的类别标签
data['class_label'] = data['class_label'].replace({2: 1})  # 将标签 2 转换为 1

# 4. 特征和标签分离
X = data.drop(['class_label'], axis=1).values  # 特征
y = data['class_label'].values  # 标签

# 5. 数据划分（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. 将标签转换为独热编码格式
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

num_classes = len(np.unique(y))  # 自动确定类别数
y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)

# ----------------------------- MLP 实现 ----------------------------- #
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

    @staticmethod
    def Activations():
        noactiv = lambda x: x
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        relu = lambda x: np.maximum(0, x)
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
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
                    loss_avg = (i / (i + 1)) * loss_avg + (1 / (i + 1)) * loss
            self.losses.append(loss_avg)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{max_iters}, Loss: {loss_avg:.4f}")

    def predict_proba(self, X_test):
        _, As = self.forward(X_test.T)
        return As[-1].T

    def predict(self, X_test):
        y_pred_prob = self.predict_proba(X_test)
        return np.argmax(y_pred_prob, axis=1)

# ----------------------------- 模型初始化与训练 ----------------------------- #
# 定义模型结构和激活函数
units = [X_train.shape[1], 128, 64, num_classes]  # 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
activations = ["relu", "relu", "softmax"]

# 初始化 MLP 模型
mlp = MLP(units, activations)

# 训练模型
mlp.fit(
    X_train, y_train_one_hot,
    lr=0.01,
    max_iters=20000,
    batch_size=32
)

# -------------------- 模型评估 -------------------- #
# 获取预测值
y_pred = mlp.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# -------------------- 绘制子图 -------------------- #
fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1,1]})

# 子图1：Loss 曲线
axs[0].plot(range(1, len(mlp.losses) + 1), mlp.losses, label="Training Loss", color='blue')
axs[0].set_title("MLP Loss Curve", fontsize=16)
axs[0].set_xlabel("Epoch", fontsize=12)
axs[0].set_ylabel("Loss", fontsize=12)
axs[0].grid(True)
axs[0].legend()

# 子图2：评估指标柱状图
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]
bars = axs[1].bar(metrics, values, color=['skyblue', 'limegreen', 'orange', 'tomato'], edgecolor='black', linewidth=1)

# 在柱形图顶部添加数值标签
for bar, value in zip(bars, values):
    axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.4f}",
                ha='center', va='bottom', fontsize=12, color='black')

# 设置子图2属性
axs[1].set_ylim(0, 1.1)
axs[1].set_title("Evaluation Metrics", fontsize=16)
axs[1].set_xlabel("Metrics", fontsize=12)
axs[1].set_ylabel("Values", fontsize=12)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)

# 调整布局
plt.tight_layout()
plt.show()