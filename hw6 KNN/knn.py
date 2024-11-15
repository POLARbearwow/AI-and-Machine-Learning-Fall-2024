import numpy as np
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

# 加载数据集
data = pd.read_csv(r"C:\Users\143\OneDrive\桌面\2024秋季\人工智能与机器学习\hw6 KNN\wdbc.data", header=None)
X = data.iloc[:, 2:].values  # 特征列（去掉ID和诊断列）
y = data.iloc[:, 1].apply(lambda x: 1 if x == 'M' else 0).values  # 将标签转换为二进制，M为1，B为0

# 数据分割：70%训练集，30%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN算法的实现
class KNN:
    def __init__(self, k=3):
        """
        初始化KNN类
        :param k: 邻居的数量
        """
        self.k = k
        self.kdtree = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练KNN模型
        :param X: 训练数据的特征，形状为(n_samples, n_features)
        :param y: 训练数据的标签，形状为(n_samples,)
        """
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def _predict(self, x):
        """
        预测单个数据点的标签
        :param x: 单个数据点的特征
        :return: 预测的标签
        """
        dist, idx = self.kdtree.query(x, k=self.k, p=2)
        
        # 检查idx是否是标量，如果是，将其转换为包含单个元素的列表
        if np.isscalar(idx):
            idx = [idx]
        
        # 获取这些邻居的标签
        neighbors_labels = [self.y_train[i] for i in idx]
        
        # 计算出现次数最多的标签
        prediction = max(set(neighbors_labels), key=neighbors_labels.count)
        return prediction

    def predict(self, X):
        """
        预测新数据点的标签
        :param X: 需要预测的数据点的特征，形状为(n_samples, n_features)
        :return: 预测的标签
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

# 测试不同的k值对准确率的影响
k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"k={k}时的准确率: {accuracy:.4f}")

# 绘制不同k值的准确率变化图
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k值")
plt.ylabel("准确率")
plt.title("不同k值下的KNN算法准确率")
plt.show()
