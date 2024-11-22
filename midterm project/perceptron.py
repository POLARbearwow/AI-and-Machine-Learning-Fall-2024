
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score

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

class MBGDPerceptron:
    def __init__(self, XTrain, learning_rate=0.01, max_iter=1000, batch_size=32):
        '''参数定义'''
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.b = 0
        self.w = np.zeros(XTrain.shape[1], dtype=np.float32)
        self.loss_history = []  # 用于记录每次迭代的损失值

    def sign(self, x, w, b):
        '''计算线性输出'''
        return np.dot(x, w) + b

    def fit(self, XTrain, yTrain):
        '''训练模型（使用 Mini-Batch Gradient Descent）'''
        n_samples = XTrain.shape[0]
        for epoch in range(self.max_iter):
            # 打乱数据顺序
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            XTrain, yTrain = XTrain[indices], yTrain[indices]

            # 计算损失
            y_pred = self.sign(XTrain, self.w, self.b)
            misclassified = np.where(yTrain * y_pred <= 0)[0]
            loss = -np.sum(yTrain[misclassified] * (np.dot(XTrain[misclassified], self.w) + self.b))
            self.loss_history.append(loss)

            # 按小批量更新权重
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = XTrain[start_idx:end_idx]
                y_batch = yTrain[start_idx:end_idx]

                # 找到错误分类的点
                y_pred = self.sign(X_batch, self.w, self.b)
                misclassified = np.where(y_batch * y_pred <= 0)[0]

                if len(misclassified) > 0:  # 有误分类点时更新
                    X_misclassified = X_batch[misclassified]
                    y_misclassified = y_batch[misclassified]

                    # 批量计算梯度
                    gradient_w = np.dot(y_misclassified, X_misclassified)
                    gradient_b = np.sum(y_misclassified)

                    # 更新权重和偏置
                    self.w += self.learning_rate * gradient_w
                    self.b += self.learning_rate * gradient_b

        return 'Perceptron with MBGD training is Done!'

    def predict(self, X):
        '''预测'''
        y = self.sign(X, self.w, self.b)
        return np.where(y > 0, 1, 0)  # 返回二分类标签

# 将目标变量转换为 +1 和 -1（感知机需要的格式）
y_train_perceptron = np.where(y_train == 1, 1, -1)
y_test_perceptron = np.where(y_test == 1, 1, -1)

# 初始化并训练模型
model = MBGDPerceptron(X_train, learning_rate=0.001, max_iter=10000, batch_size=64)
print(model.fit(X_train, y_train_perceptron))

# 使用模型进行预测
y_pred = model.predict(X_test)

# 将预测结果转回原来的标签格式（0 或 1）
y_pred_binary = np.where(y_pred > 0, 1, 0)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, zero_division=1)
recall = recall_score(y_test, y_pred_binary, zero_division=1)
f1 = f1_score(y_test, y_pred_binary)

# 输出结果
print("MBGD Perceptron 模型评估指标：")
print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
print(f"精准率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数 (F1-Score): {f1:.4f}")

# 绘制损失函数
plt.figure(figsize=(8, 6))
plt.plot(model.loss_history, label="Loss")
plt.title("Loss Function over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True)
plt.show()

# 准备数据
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

# 绘制柱形图
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])

# 在柱形图顶部添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}",
             ha='center', fontsize=10)

# 添加标题和坐标轴标签
plt.ylim(0, 1.1)  # 确保所有值都在图中显示
plt.title("Model Evaluation Metrics (Polynomial Features)", fontsize=14)
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 显示图像
plt.show()
