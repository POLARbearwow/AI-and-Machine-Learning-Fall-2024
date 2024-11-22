import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score

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

# 损失函数
def compute_loss(X, y, weights):
    predictions = X @ weights
    errors = predictions - y
    mse = (1 / (2 * len(y))) * np.sum(errors ** 2)
    return mse

# Mini-Batch Gradient Descent 实现
def mbgd_linear_regression(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    np.random.seed(42)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # 初始化权重
    losses = []  # 保存每个 epoch 的损失

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)  # 随机打乱数据
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            predictions = X_batch @ weights
            errors = predictions - y_batch
            gradient = (1 / len(y_batch)) * (X_batch.T @ errors)
            weights -= learning_rate * gradient

        # 每个 epoch 计算全局损失
        loss = compute_loss(X, y, weights)
        losses.append(loss)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
    
    return weights, losses

# 训练模型
weights_mbgd, losses = mbgd_linear_regression(X_train, y_train, learning_rate=0.01, epochs=10000, batch_size=32)

# 预测
def linear_regression_predict(X, weights):
    return X @ weights

y_pred = linear_regression_predict(X_test, weights_mbgd)
y_pred_binary = (y_pred >= 0.5).astype(int)

# 模型评估
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Precision-Recall 曲线分析
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold_pr = pr_thresholds[optimal_idx]
optimal_precision = precisions[optimal_idx]
optimal_recall = recalls[optimal_idx]
print(f"基于 Precision-Recall 曲线的最优阈值: {optimal_threshold_pr:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label="Precision-Recall Curve", color="blue")
plt.scatter(optimal_recall, optimal_precision, color="red", label=f"Optimal Threshold ({optimal_threshold_pr:.2f})", zorder=5)
plt.title("Precision-Recall Curve", fontsize=16)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.legend(loc="best", fontsize=12)
plt.grid(True)
plt.show()

# 绘制 Loss 曲线
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(losses) + 1), losses, label="Loss Curve", color="blue")
plt.title("Linear Regression Loss Curve", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss (MSE)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# 绘制四个指标的柱状图
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green", "red"])
plt.title("Linear Regression Evaluation Metrics", fontsize=16)
plt.ylabel("Score", fontsize=14)
plt.ylim(0, 1)  # 设置 y 轴范围为 0 到 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)  # 在柱子顶部标注值
plt.show()
