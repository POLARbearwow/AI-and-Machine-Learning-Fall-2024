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
    # 将特征归一化到 0-1 范围
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# 标签独热编码函数
def one_hot_encode(labels, num_classes):
    # 将标签转换为独热编码格式
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot

# 数据预处理
def preprocess_data(train_path, test_path, num_classes):
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    # 归一化数据
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
    # 独热编码标签
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test

# 数据集文件路径
train_file = './your_train_file.csv'  # 替换为你的训练数据路径
test_file = './your_test_file.csv'    # 替换为你的测试数据路径

# 定义类别数
num_classes = 3  # 替换为你的数据集类别数量

# 调用数据预处理函数
X_train, y_train_one_hot, X_test, y_test_one_hot, y_train, y_test = preprocess_data(train_file, test_file, num_classes)

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

# ------------ 模型初始化与训练 ------------

# 定义模型结构和激活函数
units = [X_train.shape[1], 128, 64, num_classes]  # 输入层, 隐藏层, 输出层
activations = ["relu", "relu", "softmax"]

# 初始化 MLP 模型
mlp = MLP(units, activations)

# 训练模型
mlp.fit(
    X_train, y_train_one_hot,
    lr=0.01,
    max_iters=1000,
    batch_size=32
)

# 测试模型
y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
