import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score

# ---------------------------------- 数据预处理 ---------------------------------
# 数据加载和预处理
data = pd.read_csv('./midterm project/ai4i2020.csv')
data['Type'] = data['Type'].astype('category').cat.codes
data = data.drop(['UDI', 'Product ID'], axis=1)

features_to_scale = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

X = data.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1).values  # 特征
y = data['Machine failure'].values  # 标签

# 统计正负样本数量
value_counts = pd.Series(y).value_counts()
N_pos = value_counts[1]  # 正样本数量
N_neg = value_counts[0]  # 负样本数量

# 数据集划分
def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ---------------------------------- 模型定义 ---------------------------------
class LogisticRegressionMiniBatch:
    def __init__(self, n_features, lr=0.1, n_iter=10000, batch_size=32, tol=1e-10, patience=10):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.weights = np.random.randn(n_features) * 0.05
        self.losses = []
        self.w1 = None  # 正样本权重
        self.w2 = None  # 负样本权重

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y_true, y_pred_prob):
        '''加权交叉熵损失'''
        epsilon = 1e-10  # 防止 log(0)
        y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        return -np.mean(
            self.w1 * y_true * np.log(y_pred_prob) + 
            self.w2 * (1 - y_true) * np.log(1 - y_pred_prob)
        )

    def _gradient(self, X, y_true, y_pred_prob):
        errors = y_pred_prob - y_true
        grad = np.dot(X.T, errors) / X.shape[0]
        return grad

    def _compute_sample_weights(self, y):
        '''计算正负样本的权重'''
        self.w1 = 1 / N_pos
        self.w2 = 1 / N_neg

    def fit(self, X_train, y_train):
        '''训练模型'''
        self._compute_sample_weights(y_train)  # 计算正负样本权重
        for epoch in range(self.n_iter):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)  # 随机打乱数据
            X_train, y_train = X_train[indices], y_train[indices]
            
            epoch_loss = 0
            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]
                
                y_pred_prob = self._sigmoid(np.dot(X_batch, self.weights))
                batch_loss = self.loss(y_batch, y_pred_prob)
                epoch_loss += batch_loss * len(y_batch)  # 累加批量损失

                grad = self._gradient(X_batch, y_batch, y_pred_prob)
                self.weights -= self.lr * grad  # 更新权重

            epoch_loss /= X_train.shape[0]  # 计算平均损失
            self.losses.append(epoch_loss)

    def predict_proba(self, X_test):
        '''预测概率'''
        y_pred_prob = self._sigmoid(np.dot(X_test, self.weights))
        return y_pred_prob

    def predict(self, X_test, threshold=0.5):
        '''预测二分类标签'''
        y_pred_prob = self.predict_proba(X_test)
        y_pred = np.where(y_pred_prob >= threshold, 1, 0)
        return y_pred

# ---------------------------------- 特征扩展 ---------------------------------
degree = 3  # 多项式扩展阶数
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 更新特征数量
n_features_poly = X_train_poly.shape[1]

# 创建并训练模型
model = LogisticRegressionMiniBatch(n_features=n_features_poly, lr=0.0001, n_iter=10000, batch_size=64)
model.fit(X_train_poly, y_train)

# 预测和评估
y_pred = model.predict(X_test_poly, threshold=0.24)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ---------------------------------- 模型评估与绘图 ---------------------------------

# 评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# -------------------- 绘制损失函数曲线 --------------------
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(model.losses) + 1), model.losses, label="Loss")
plt.title("Logistic Regression Loss Function with Polynomial Features", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# -------------------- Precision-Recall 曲线 --------------------
# 获取预测概率
y_pred_proba = model.predict_proba(X_test_poly)

# 计算 Precision-Recall 曲线
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# 计算 F1 分数并找到最优阈值
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # 防止除零
optimal_idx = np.argmax(f1_scores[:-1])  # 去掉最后一个点
optimal_threshold_pr = thresholds[optimal_idx]

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(10, 8))
plt.plot(recalls, precisions, label="Precision-Recall Curve", linewidth=2, color="blue")
plt.scatter(recalls[optimal_idx], precisions[optimal_idx], color="red", label=f"Optimal Threshold = {optimal_threshold_pr:.2f}", s=100)
plt.title("Precision-Recall Curve", fontsize=18)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print(f"Optimal Threshold based on Precision-Recall: {optimal_threshold_pr}")

# -------------------- 绘制柱形图 --------------------
# 准备数据
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

# 绘制柱形图
plt.figure(figsize=(10, 8))
bars = plt.bar(metrics, values, color=['skyblue', 'limegreen', 'orange', 'tomato'], edgecolor='black', linewidth=1)

# 在柱形图顶部添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}",
             ha='center', va='bottom', fontsize=12, color='black')

# 添加标题和坐标轴标签
plt.ylim(0, 1.1)  # 确保所有值都在图中显示
plt.title("Logistic Regression Evaluation Metrics with Polynomial Features", fontsize=18)
plt.xlabel("Metrics", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 显示图像
plt.show()
