import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams['font.sans-serif'] = ['SimHei']  
# rcParams['axes.unicode_minus'] = False   


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_iters=1000):
        """
        :param learning_rate: 学习率
        :param lambda_param: 正则化参数
        :param max_iters: 最大迭代次数
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.losses = []  # 用于记录每次迭代的损失

    def fit(self, X, y):
        """
        训练 SVM 模型
        :param X: 输入特征 (m x n)
        :param y: 标签 (-1 或 1)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y = np.where(y <= 0, -1, 1)  # 将标签从 0, 1 转换为 -1, 1

        for _ in range(self.max_iters):
            loss = 0
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    loss += 1 - y[idx] * (np.dot(x_i, self.w) + self.b)  # 合页损失

            # 正则化项
            loss += self.lambda_param * np.sum(self.w ** 2)
            self.losses.append(loss)

    def predict(self, X):
        """
        预测标签
        :param X: 输入特征
        :return: 预测标签
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# ----------------------------- 数据加载与预处理 ----------------------------- #
# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 分离特征和目标值
features = data.iloc[:, :-1].values  # 除最后一列外的特征数据
target = data.iloc[:, -1].values  # 最后一列为目标值

# 4. 归一化
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

# 将目标值转换为二分类问题（以 1 类和其他类为例，构建二分类问题）
binary_target = np.where(target == 1, 1, 0)

# 5. 数据划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_features, binary_target, test_size=0.3, random_state=42)

# ----------------------------- SVM 训练 ----------------------------- #
# 创建 SVM 模型实例
svm = SVM(learning_rate=0.001, lambda_param=0.01, max_iters=10000)
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# ----------------------------- 评估模型 ----------------------------- #
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("SVM 模型评估结果：")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 得分 (F1 Score): {f1:.4f}")

# ----------------------------- Visualize Decision Boundary and Loss Curve ----------------------------- #
def plot_results(X, y, model, losses):
    """
    Plot SVM decision boundary and loss curve
    :param X: Input features (only 2D features for visualization)
    :param y: Labels
    :param model: Trained SVM model
    :param losses: List of loss values
    """
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Decision Boundary
    x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01), np.arange(x1_min, x1_max, 0.01))
    grid = np.c_[xx0.ravel(), xx1.ravel()]

    # Add extra dimensions if necessary to match model weights
    if X.shape[1] < model.w.shape[0]:
        additional_dims = np.zeros((grid.shape[0], model.w.shape[0] - X.shape[1]))
        grid = np.hstack((grid, additional_dims))

    grid_predictions = model.predict(grid)
    grid_predictions = grid_predictions.reshape(xx0.shape)

    axs[0].contourf(xx0, xx1, grid_predictions, alpha=0.8, cmap=plt.cm.coolwarm)
    axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    axs[0].set_title("SVM Decision Boundary", fontsize=14)
    axs[0].set_xlabel("Feature 1", fontsize=12)
    axs[0].set_ylabel("Feature 2", fontsize=12)

    # Subplot 2: Loss Curve
    axs[1].plot(range(len(losses)), losses, label='Training Loss', color='blue')
    axs[1].set_title("Loss Function Over Iterations", fontsize=14)
    axs[1].set_xlabel("Iterations", fontsize=12)
    axs[1].set_ylabel("Loss Value", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Call the visualization function
plot_results(X_test[:, :2], y_test, svm, svm.losses)

