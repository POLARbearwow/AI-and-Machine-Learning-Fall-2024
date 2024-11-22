import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
import math
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

"----------------------------------data preprocessing-------------------------------------------------"
#region data preprocessing

# print("当前工作目录:",os.getcwd())
data = pd.read_csv('./midterm project/ai4i2020.csv')
data['Type'] = data['Type'].astype('category').cat.codes

# features_only = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
#                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
# filtered_data = data[features_only]

# # 计算相关性矩阵，并只保留与 Machine failure 的相关性
# correlation_with_target = filtered_data.corr()[['Machine failure']]

# # 绘制热力图
# plt.figure(figsize=(6, 8))
# sns.heatmap(correlation_with_target, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1)
# plt.title('Correlation with Machine Failure')
# plt.show()

# 删除无关特征 UDI 和 Product ID
data = data.drop(['UDI', 'Product ID'], axis=1)

# 标准化特征列 
features_to_scale = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

X = data.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)  # 特征
y = data['Machine failure']  # 标签

# # 检查数据集中的缺失值
# print(data.isnull().sum())
# print(data.head())
# print(X.shape[0])

# # 提取特征和目标变量
# X = data.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)  # 特征
# y = data['Machine failure']  # 标签

# # 训练随机森林模型
# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)

# # 使用 PartialDependenceDisplay 替代 plot_partial_dependence
# features_to_plot = [0, 1,2,3,4,5]  # 特征索引，比如第一个和第二个特征
# PartialDependenceDisplay.from_estimator(model, X, features=features_to_plot, feature_names=X.columns, grid_resolution=50)

# # 展示图表
# plt.show()


# 检查处理后的数据集
# print(data.head(10))
# print(X.head(10))
# print(y.head(10))

X = X.values
y = y.values

X_regression =  np.hstack((np.ones((X.shape[0], 1)), X))  # 添加偏置项

def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx =  indices[:split], indices[split:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X_train[:5])

#endregion

value_counts = pd.Series(y).value_counts()
N_pos = value_counts[1]  # 正样本数量
N_neg = value_counts[0]  # 负样本数量

# 计算比例
positive_ratio = N_pos / len(y)  # 正样本比例
negative_ratio = N_neg / len(y)  # 负样本比例

# print(f"正样本数量: {N_pos}, 负样本数量: {N_neg}")
# print(f"正样本比例: {positive_ratio:.4f}, 负样本比例: {negative_ratio:.4f}")

"----------------------------------linear regression-------------------------------------------------"
#region linear regression
# def linear_regression_train(X, y):
#     """
#     解析解法：w = (X^T * X)^(-1) * X^T * y
#     """
#     X_transpose = X.T
#     w = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
#     return w

# # 训练模型，得到权重参数
# weights = linear_regression_train(X_train, y_train)

# def linear_regression_predict(X, weights):
#     return X @ weights

# y_pred = linear_regression_predict(X_test, weights)

# # 将预测值转换为二分类标签    threshold是linear regression做分类的全部了……
# y_pred_binary = (y_pred >= 0.15).astype(int)

# # 使用 sklearn 计算分类指标
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = f1_score(y_test, y_pred_binary)

# # 输出结果
# print("分类模型评估指标：")
# print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
# print(f"精准率 (Precision): {precision:.4f}")
# print(f"召回率 (Recall): {recall:.4f}")
# print(f"F1 分数 (F1-Score): {f1:.4f}")

# # 计算 Precision-Recall 曲线
# precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred)
# f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
# optimal_idx = np.argmax(f1_scores)
# optimal_threshold_pr = pr_thresholds[optimal_idx]
# optimal_precision = precisions[optimal_idx]
# optimal_recall = recalls[optimal_idx]
# print(f"基于 Precision-Recall 曲线的最优阈值: {optimal_threshold_pr:.2f}")

# # 绘制 Precision-Recall 曲线
# plt.figure(figsize=(8, 6))
# plt.plot(recalls, precisions, label="Precision-Recall Curve", color="blue")
# plt.scatter(optimal_recall, optimal_precision, color="red", label=f"Optimal Threshold ({optimal_threshold_pr:.2f})", zorder=5)
# plt.title("Precision-Recall Curve", fontsize=16)
# plt.xlabel("Recall", fontsize=14)
# plt.ylabel("Precision", fontsize=14)
# plt.legend(loc="best", fontsize=12)
# plt.grid(True)
# plt.show()
# #endregion

"--------------------------------------perceptron----------------------------------------------------"
#region perceptron

# class MBGDPerceptron:
#     def __init__(self, XTrain, learning_rate=0.01, max_iter=1000, batch_size=32):
#         '''参数定义'''
#         self.learning_rate = learning_rate
#         self.max_iter = max_iter
#         self.batch_size = batch_size
#         self.b = 0
#         self.w = np.zeros(XTrain.shape[1], dtype=np.float32)
#         self.loss_history = []  # 用于记录每次迭代的损失值

#     def sign(self, x, w, b):
#         '''计算线性输出'''
#         return np.dot(x, w) + b

#     def fit(self, XTrain, yTrain):
#         '''训练模型（使用 Mini-Batch Gradient Descent）'''
#         n_samples = XTrain.shape[0]
#         for epoch in range(self.max_iter):
#             # 打乱数据顺序
#             indices = np.arange(n_samples)
#             np.random.shuffle(indices)
#             XTrain, yTrain = XTrain[indices], yTrain[indices]

#             # 计算损失
#             y_pred = self.sign(XTrain, self.w, self.b)
#             misclassified = np.where(yTrain * y_pred <= 0)[0]
#             loss = -np.sum(yTrain[misclassified] * (np.dot(XTrain[misclassified], self.w) + self.b))
#             self.loss_history.append(loss)

#             # 按小批量更新权重
#             for start_idx in range(0, n_samples, self.batch_size):
#                 end_idx = start_idx + self.batch_size
#                 X_batch = XTrain[start_idx:end_idx]
#                 y_batch = yTrain[start_idx:end_idx]

#                 # 找到错误分类的点
#                 y_pred = self.sign(X_batch, self.w, self.b)
#                 misclassified = np.where(y_batch * y_pred <= 0)[0]

#                 if len(misclassified) > 0:  # 有误分类点时更新
#                     X_misclassified = X_batch[misclassified]
#                     y_misclassified = y_batch[misclassified]

#                     # 批量计算梯度
#                     gradient_w = np.dot(y_misclassified, X_misclassified)
#                     gradient_b = np.sum(y_misclassified)

#                     # 更新权重和偏置
#                     self.w += self.learning_rate * gradient_w
#                     self.b += self.learning_rate * gradient_b

#         return 'Perceptron with MBGD training is Done!'

#     def predict(self, X):
#         '''预测'''
#         y = self.sign(X, self.w, self.b)
#         return np.where(y > 0, 1, 0)  # 返回二分类标签

# # 将目标变量转换为 +1 和 -1（感知机需要的格式）
# y_train_perceptron = np.where(y_train == 1, 1, -1)
# y_test_perceptron = np.where(y_test == 1, 1, -1)

# # 初始化并训练模型
# model = MBGDPerceptron(X_train, learning_rate=0.001, max_iter=1000, batch_size=64)
# print(model.fit(X_train, y_train_perceptron))

# # 使用模型进行预测
# y_pred = model.predict(X_test)

# # 将预测结果转回原来的标签格式（0 或 1）
# y_pred_binary = np.where(y_pred > 0, 1, 0)

# # 评估模型性能
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary, zero_division=1)
# recall = recall_score(y_test, y_pred_binary, zero_division=1)
# f1 = f1_score(y_test, y_pred_binary)

# # 输出结果
# print("MBGD Perceptron 模型评估指标：")
# print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
# print(f"精准率 (Precision): {precision:.4f}")
# print(f"召回率 (Recall): {recall:.4f}")
# print(f"F1 分数 (F1-Score): {f1:.4f}")

# # 绘制损失函数
# plt.figure(figsize=(8, 6))
# plt.plot(model.loss_history, label="Loss")
# plt.title("Loss Function over Epochs", fontsize=16)
# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("Loss", fontsize=14)
# plt.legend(loc="best", fontsize=12)
# plt.grid(True)
# plt.show()

#endregion

"--------------------------------------perceptron(polynomial)----------------------------------------------------"
#region perceptron(polynomial)
# class MBGDPerceptron:
#     def __init__(self, XTrain, learning_rate=0.001, max_iter=10000, batch_size=32):
#         '''参数定义'''
#         self.learning_rate = learning_rate
#         self.max_iter = max_iter
#         self.batch_size = batch_size
#         self.b = 0
#         self.w = np.zeros(XTrain.shape[1], dtype=np.float32)
#         self.loss_history = []  # 用于记录每次迭代的损失值

#     def sign(self, x, w, b):
#         '''计算线性输出'''
#         return np.dot(x, w) + b

#     def fit(self, XTrain, yTrain):
#         '''训练模型（使用 Mini-Batch Gradient Descent）'''
#         n_samples = XTrain.shape[0]
#         for epoch in range(self.max_iter):
#             # 随机打乱数据顺序
#             indices = np.arange(n_samples)
#             np.random.shuffle(indices)
#             XTrain, yTrain = XTrain[indices], yTrain[indices]

#             # 计算损失
#             y_pred = self.sign(XTrain, self.w, self.b)
#             misclassified = np.where(yTrain * y_pred <= 0)[0]
#             loss = -np.sum(yTrain[misclassified] * (np.dot(XTrain[misclassified], self.w) + self.b))
#             self.loss_history.append(loss)

#             # 按小批量更新权重
#             for start_idx in range(0, n_samples, self.batch_size):
#                 end_idx = start_idx + self.batch_size
#                 X_batch = XTrain[start_idx:end_idx]
#                 y_batch = yTrain[start_idx:end_idx]

#                 # 找到当前批量中的误分类点
#                 y_pred_batch = self.sign(X_batch, self.w, self.b)
#                 misclassified_batch = np.where(y_batch * y_pred_batch <= 0)[0]

#                 if len(misclassified_batch) > 0:  # 如果有误分类点，更新权重和偏置
#                     X_misclassified = X_batch[misclassified_batch]
#                     y_misclassified = y_batch[misclassified_batch]

#                     # 计算梯度
#                     gradient_w = np.dot(y_misclassified, X_misclassified)
#                     gradient_b = np.sum(y_misclassified)

#                     # 更新参数
#                     self.w += self.learning_rate * gradient_w
#                     self.b += self.learning_rate * gradient_b

#         return 'Perceptron with MBGD training is Done!'

#     def predict(self, X):
#         '''预测'''
#         y = self.sign(X, self.w, self.b)
#         return np.where(y > 0, 1, 0)  # 返回二分类标签

# # -------------------- 数据处理部分 --------------------
# # 读取数据和预处理（假设 X_train, X_test, y_train, y_test 已经从前面得到）

# # 生成多项式特征
# poly = PolynomialFeatures(degree=3, include_bias=False)  # 二次多项式特征扩展
# X_train_poly = poly.fit_transform(X_train)  # 对训练数据扩展特征
# X_test_poly = poly.transform(X_test)        # 对测试数据扩展特征

# # 将目标变量转换为感知机需要的格式 (+1 和 -1)
# y_train_perceptron = np.where(y_train == 1, 1, -1)
# y_test_perceptron = np.where(y_test == 1, 1, -1)

# # -------------------- 模型训练和预测部分 --------------------
# # 初始化并训练感知机模型
# model = MBGDPerceptron(X_train_poly, learning_rate=0.001, max_iter=2000, batch_size=64)
# print(model.fit(X_train_poly, y_train_perceptron))

# # 使用模型进行预测
# y_pred = model.predict(X_test_poly)

# # 将预测结果转回原来的标签格式（0 或 1）
# y_pred_binary = np.where(y_pred > 0, 1, 0)

# # -------------------- 评估模型性能 --------------------
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary, zero_division=1)
# recall = recall_score(y_test, y_pred_binary, zero_division=1)
# f1 = f1_score(y_test, y_pred_binary)

# # 输出结果
# print("MBGD Perceptron 模型 (多项式特征) 评估指标：")
# print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
# print(f"精准率 (Precision): {precision:.4f}")
# print(f"召回率 (Recall): {recall:.4f}")
# print(f"F1 分数 (F1-Score): {f1:.4f}")

# # -------------------- 绘制损失函数 --------------------
# plt.figure(figsize=(8, 6))
# plt.plot(model.loss_history, label="Loss")
# plt.title("Loss Function over Epochs (Polynomial Features)", fontsize=16)
# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("Loss", fontsize=14)
# plt.legend(loc="best", fontsize=12)
# plt.grid(True)
# plt.show()
# #endregion    

"--------------------------------------logistic regression ----------------------------------------------------"
#region logistic regression

# class LogisticRegressionMiniBatch:
#     def __init__(self, n_features, lr=0.1, n_iter=10000, batch_size=32, tol=1e-10, patience=10):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.batch_size = batch_size
#         self.tol = tol
#         self.patience = patience
#         self.weights = np.random.randn(n_features + 1) * 0.05
#         self.losses = []
#         self.w1 = None  # 正样本权重
#         self.w2 = None  # 负样本权重

#     def _sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def loss(self, y_true, y_pred_prob):
#         '''加权交叉熵损失'''
#         epsilon = 1e-10  # 防止 log(0)
#         y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
#         return -np.mean(
#             self.w1 * y_true * np.log(y_pred_prob) + 
#             self.w2 * (1 - y_true) * np.log(1 - y_pred_prob)
#         )

#     def _gradient(self, X, y_true, y_pred_prob):
#         errors = y_pred_prob - y_true
#         grad = np.dot(X.T, errors) / X.shape[0]
#         return grad

#     def _preprocess_data(self, X):
#         m, n = X.shape
#         X = np.hstack([np.ones((m, 1)), X])  # 添加偏置项
#         return X

#     def _update_weights(self, X, y):
#         '''更新权重并训练模型'''
#         no_improvement = 0
#         for epoch in range(self.n_iter):
#             indices = np.arange(X.shape[0])
#             np.random.shuffle(indices)  # 随机打乱数据
#             X, y = X[indices], y[indices]
            
#             epoch_loss = 0
#             for start in range(0, X.shape[0], self.batch_size):
#                 end = start + self.batch_size
#                 X_batch, y_batch = X[start:end], y[start:end]
                
#                 y_pred_prob = self._sigmoid(np.dot(X_batch, self.weights))
#                 batch_loss = self.loss(y_batch, y_pred_prob)
#                 epoch_loss += batch_loss * len(y_batch)  # 累加批量损失

#                 grad = self._gradient(X_batch, y_batch, y_pred_prob)
#                 self.weights -= self.lr * grad  # 更新权重

#             epoch_loss /= X.shape[0]  # 计算平均损失
#             self.losses.append(epoch_loss)
            
#             if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
#                 no_improvement += 1
#                 if no_improvement >= self.patience:
#                     print(f"Early stopping triggered at epoch {epoch}.")
#                     break
#             else:
#                 no_improvement = 0

#     def _compute_sample_weights(self, y):
#         '''计算正负样本的权重'''
#         self.w1 = 1 / N_pos
#         self.w2 = 1 / N_neg

#     def fit(self, X_train, y_train):
#         '''训练模型'''
#         X_train = self._preprocess_data(X_train)  # 添加偏置项
#         self._compute_sample_weights(y_train)  # 计算正负样本权重
#         self._update_weights(X_train, y_train)  # 更新权重

#     def predict_proba(self, X_test):
#         '''预测概率'''
#         X_test = self._preprocess_data(X_test)
#         y_pred_prob = self._sigmoid(np.dot(X_test, self.weights))
#         return y_pred_prob

#     def predict(self, X_test, threshold=0.5):
#         '''预测二分类标签'''
#         y_pred_prob = self.predict_proba(X_test)
#         y_pred = np.where(y_pred_prob >= threshold, 1, 0)
#         return y_pred
# #region    
# # class LogisticRegressionMBGD:
# #     def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32):
# #         self.learning_rate = learning_rate
# #         self.max_iter = max_iter
# #         self.batch_size = batch_size
# #         self.theta = None
# #         self.loss_history = []
# #         self.w1 = 1 / N_pos  # 正样本权重
# #         self.w2 = 1 / N_neg  # 负样本权重

# #     def sigmoid(self, z):
# #         '''Sigmoid 函数'''
# #         return 1 / (1 + np.exp(-z))

# #     def compute_loss(self, h, y, w1, w2):
# #         '''计算加权交叉熵损失'''
# #         m = len(y)
# #         # 根据正负样本权重，计算加权交叉熵
# #         loss = -1 / m * np.sum(
# #             self.w1 * y * np.log(h + 1e-9) + self.w2 * (1 - y) * np.log(1 - h + 1e-9)
# #         )
# #         return loss

# #     def fit(self, X, y):
# #         '''训练模型（使用加权交叉熵的 MBGD）'''
# #         m, n = X.shape
# #         self.theta = np.zeros(n)  # 初始化参数
# #         self.loss_history = []

# #         # 计算正负样本数量和权重


# #         for epoch in range(self.max_iter):
# #             # 随机打乱数据
# #             indices = np.arange(m)
# #             np.random.shuffle(indices)
# #             X = X[indices]
# #             y = y[indices]

# #             for start_idx in range(0, m, self.batch_size):
# #                 end_idx = start_idx + self.batch_size
# #                 X_batch = X[start_idx:end_idx]
# #                 y_batch = y[start_idx:end_idx]

# #                 # 计算预测值
# #                 z = np.dot(X_batch, self.theta)
# #                 h = self.sigmoid(z)

# #                 # 计算梯度
# #                 gradient = np.dot(X_batch.T, (h - y_batch)) / len(y_batch)

# #                 # 更新参数
# #                 self.theta -= self.learning_rate * gradient

# #             # 计算全数据集的加权损失
# #             z = np.dot(X, self.theta)
# #             h = self.sigmoid(z)
# #             loss = self.compute_loss(h, y, self.w1, self.w2)
# #             self.loss_history.append(loss)

# #         return self

# #     def predict_proba(self, X):
# #         '''计算概率'''
# #         z = np.dot(X, self.theta)
# #         return self.sigmoid(z)

# #     def predict(self, X, threshold=0.5):
# #         '''预测二分类标签'''
# #         probabilities = self.predict_proba(X)
# #         return (probabilities >= threshold).astype(int)
# #endregion



# # -------------------- Logistic Regression 训练 --------------------
# # 确定特征数
# n_features = X_train.shape[1]

# # 初始化并训练模型
# model = LogisticRegressionMiniBatch(
#     n_features=n_features, 
#     lr=0.0000001, 
#     n_iter=9000, 
#     batch_size=32, 
#     tol=1e-20, 
#     patience=10
# )
# model.fit(X_train, y_train)

# # -------------------- 评估模型 --------------------
# # 使用模型进行预测
# y_pred = model.predict(X_test)

# # 计算准确率、精准率、召回率和 F1 分数
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, zero_division=1)
# recall = recall_score(y_test, y_pred, zero_division=1)
# f1 = f1_score(y_test, y_pred)

# print("Logistic Regression with Mini-Batch Gradient Descent 模型评估指标：")
# print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
# print(f"精准率 (Precision): {precision:.4f}")
# print(f"召回率 (Recall): {recall:.4f}")
# print(f"F1 分数 (F1-Score): {f1:.4f}")

# # -------------------- 绘制损失函数 --------------------
# plt.figure(figsize=(8, 6))
# plt.plot(model.losses, label="Loss")
# plt.title("Loss Function over Iterations (MBGD)", fontsize=16)
# plt.xlabel("Iteration", fontsize=14)
# plt.ylabel("Loss", fontsize=14)
# plt.legend(loc="best", fontsize=12)
# plt.grid(True)
# plt.show()

# # -------------------- Precision-Recall 曲线 --------------------
# # 获取预测概率
# y_pred_proba = model.predict_proba(X_test)

# # 计算 Precision-Recall 曲线
# precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# # 计算 F1 分数，并找到最优阈值
# f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # 防止除零
# optimal_idx = np.argmax(f1_scores[:-1])  # 去掉最后一个 F1 分数对应点
# optimal_threshold_pr = thresholds[optimal_idx]

# # 绘制 Precision-Recall 曲线
# plt.figure(figsize=(8, 6))
# plt.plot(recalls, precisions, label="Precision-Recall Curve")
# plt.scatter(recalls[optimal_idx], precisions[optimal_idx], color='red', label=f"Optimal Threshold = {optimal_threshold_pr:.2f}")
# plt.title("Precision-Recall Curve", fontsize=16)
# plt.xlabel("Recall", fontsize=14)
# plt.ylabel("Precision", fontsize=14)
# plt.legend(loc="best", fontsize=12)
# plt.grid(True)
# plt.show()

# print(f"Optimal Threshold based on Precision-Recall: {optimal_threshold_pr}")
#endregion

"---------------------------------------------MLP ----------------------------------------------------"

# class MLP:
#     def __init__(self, units, activs, w1, w2):
#         self.units = units
#         self.length = len(units)
#         self.activations = activs
#         self.w1 = w1  # 正样本的权重
#         self.w2 = w2  # 负样本的权重
#         self.losses = []  # 用于记录每个 epoch 的平均损失
#         assert len(units) - 1 == len(activs) and set(activs).issubset(
#             set(["noactiv", "relu", "sigmoid", "softmax", "tanh"])
#         ) and "softmax" not in activs[:-1]
#         activDict, derivDict = MLP.Activations()
#         self.activs = [None] + [activDict[i] for i in activs]
#         self.derivs = [None] + [derivDict[i] if i != "softmax" else None for i in activs]
#         self.Ws = [None] + [np.random.randn(units[i+1], units[i]) * 0.01 for i in range(len(units) - 1)]
#         self.bs = [None] + [np.zeros((units[i+1], 1)) for i in range(len(units) - 1)]

#     def Activations():
#         noactiv = lambda x: x
#         sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         relu = lambda x: np.maximum(0, x)
#         softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
#         tanh = lambda x: np.tanh(x)
#         noactiv_d = lambda x: np.ones_like(x)
#         sigmoid_d = lambda x: sigmoid(x) * (1 - sigmoid(x))
#         relu_d = lambda x: (x > 0).astype(float)
#         tanh_d = lambda x: 1 - tanh(x)**2
#         activations = {"noactiv": noactiv, "sigmoid": sigmoid, "relu": relu, "softmax": softmax, "tanh": tanh}
#         derivatives = {"noactiv": noactiv_d, "sigmoid": sigmoid_d, "relu": relu_d, "tanh": tanh_d}
#         return activations, derivatives

#     def forward(self, X):
#         Zs, As = [None] * self.length, [None] * self.length
#         As[0] = X
#         for i in range(1, self.length):
#             Zs[i] = self.Ws[i] @ As[i-1] + self.bs[i]
#             As[i] = self.activs[i](Zs[i])
#         return Zs, As

#     def _weighted_cross_entropy_loss(self, y_true, y_pred):
#         epsilon = 1e-10
#         y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
#         return -np.mean(
#             self.w1 * y_true * np.log(y_pred) + self.w2 * (1 - y_true) * np.log(1 - y_pred)
#         )

#     def backward(self, X, Y, lr):
#         amount = X.shape[-1]
#         Zs, As = self.forward(X)
#         loss = self._weighted_cross_entropy_loss(Y, As[-1])
#         if self.activations[-1] == "softmax":
#             dZ = As[-1] - Y
#         else:
#             dA = -(self.w1 * Y / As[-1] - self.w2 * (1 - Y) / (1 - As[-1]))
#         for l in range(self.length - 1, 0, -1):
#             if self.activations[-1] != "softmax" or l < self.length - 1:
#                 dZ = dA * self.derivs[l](Zs[l])
#             dW = 1 / amount * dZ @ As[l-1].T
#             db = 1 / amount * np.sum(dZ, axis=1, keepdims=True)
#             dA = self.Ws[l].T @ dZ
#             self.Ws[l] -= lr * dW
#             self.bs[l] -= lr * db
#         return loss

#     def fit(self, X, Y, lr, max_iters, batch_size=None):
#         assert X.shape[-1] == self.units[0]
#         X, Y = X.T, Y.reshape(-1, 1).T
#         amount = X.shape[-1]
#         for epoch in range(max_iters):
#             if not batch_size:
#                 loss_avg = self.backward(X, Y, lr)
#             else:
#                 loss_avg = 0
#                 for i in range(math.ceil(amount / batch_size)):
#                     loss = self.backward(
#                         X[:, i*batch_size:(i+1)*batch_size],
#                         Y[:, i*batch_size:(i+1)*batch_size],
#                         lr
#                     )
#                     loss_avg = (i / (i+1)) * loss_avg + (1 / (i+1)) * loss
#             self.losses.append(loss_avg)  # 保存每个 epoch 的损失值
#             print(f"Epoch {epoch+1}/{max_iters}, Loss: {loss_avg}")

#     def predict_proba(self, X_test):
#         _, As = self.forward(X_test.T)
#         return As[-1].T

#     def predict(self, X_test, threshold=0.5):
#         y_pred_prob = self.predict_proba(X_test)
#         y_pred = (y_pred_prob >= threshold).astype(int)
#         return y_pred

# # ------------ 模型初始化与训练 ------------
# units = [X_train.shape[1], 16, 8, 1]
# activations = ["sigmoid", "sigmoid", "sigmoid"]
# mlp = MLP(units, activations, N_pos, N_neg)

# mlp.fit(
#     X_train, y_train,
#     lr=0.00001,
#     max_iters=200000,
#     batch_size=32
# )

# # ------------ 绘制损失函数 ------------
# mlp_structure = "MLP Structure: Input Layer -> 16 Nodes (Hidden) -> 8 Nodes (Hidden) -> 1 Node (Output)"


# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(mlp.losses) + 1), mlp.losses, label="Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(f"MLP loss\n{mlp_structure}", fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.show()

# # 计算评估指标
# y_pred_proba = mlp.predict_proba(X_test)
# y_pred = mlp.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# # MLP 的层次结构
# mlp_structure = "MLP Structure: Input Layer -> 16 Nodes (Hidden) -> 8 Nodes (Hidden) -> 1 Node (Output)"

# # 准备数据
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
# values = [accuracy, precision, recall, f1]  # 确保使用计算结果变量，而不是函数名

# # 绘制柱状图
# plt.figure(figsize=(10, 6))
# plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
# plt.ylim(0, 1.1)
# plt.title(f"Loss Curve (MLP)\n{mlp_structure}", fontsize=14)
# plt.ylabel("Metric Value")
# plt.xlabel("Metrics")

# # 显示每个柱状图顶部的数值
# for i, v in enumerate(values):
#     plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

# # 添加网格线
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # 显示图像
# plt.show()