#有没有参数 DecisionStump？ loss怎么计算的

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# 1. 读取数据
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns


# 3. 将 class_label 转换为 0, 2 的格式 (class的shi'j)
data = data[data['class_label'] != 2]
data['class_label'] = data['class_label'] - 1  # 转换为从 0 开始的类别标签
data['class_label'] = data['class_label'].replace({2: 1})  # 将标签 2 转换为 1



# 4. 特征和标签分离
X = data.drop(['class_label'], axis=1).values  # 特征
y = data['class_label'].values  # 标签

# 5. 数据划分（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. 定义弱分类器（决策树桩）
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None
    
    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        # 遍历所有特征和可能的阈值，寻找最佳分割点
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = self._predict_stump(X[:, feature_i], threshold, polarity)
                    error = np.sum(sample_weights * (predictions != y))
                    
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_i
                        self.threshold = threshold
                        self.polarity = polarity

    def _predict_stump(self, X_column, threshold, polarity):
        predictions = np.ones(X_column.shape)
        if polarity == 1:
            predictions[X_column < threshold] = -1  # 预测为负类
        else:
            predictions[X_column >= threshold] = -1
        return predictions

    def predict(self, X):
        X_column = X[:, self.feature_index]
        return self._predict_stump(X_column, self.threshold, self.polarity)

# 8. 定义 AdaBoost 算法
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []
        self.losses = []  # 用于记录每次迭代的损失值
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, sample_weights)
            predictions = stump.predict(X)
            
            # 计算分类误差
            error = np.sum(sample_weights * (predictions != y)) + 1e-10
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.stumps.append(stump)
            
            # 更新样本权重
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)
            
            # 计算并记录当前的指数损失
            f_x = self._calculate_f(X)
            loss = np.sum(np.exp(-y * f_x))
            self.losses.append(loss)
    
    def _calculate_f(self, X):
        """计算加权分类器的输出值 f(x)"""
        f_x = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            f_x += alpha * stump.predict(X)
        return f_x
    
    def predict(self, X):
        f_x = self._calculate_f(X)
        return np.sign(f_x).astype(int)

# 9. 将多分类问题转换为二分类问题（以分类0 vs 其他为例）
y_train_binary = np.where(y_train == 0, 1, -1)
y_test_binary = np.where(y_test == 0, 1, -1)

# 检查类别分布
print("Training class distribution:", Counter(y_train_binary))
print("Test class distribution:", Counter(y_test_binary))

# 10. 训练 AdaBoost 模型
adaboost = AdaBoost(n_estimators=50)
adaboost.fit(X_train, y_train_binary)

# 11. 预测
y_pred_train = adaboost.predict(X_train)
y_pred_test = adaboost.predict(X_test)

# 12. 检查预测值分布
print("Train predictions distribution:", Counter(y_pred_train))
print("Test predictions distribution:", Counter(y_pred_test))

# 13. 评估模型性能
train_accuracy = accuracy_score(y_train_binary, y_pred_train)
test_accuracy = accuracy_score(y_test_binary, y_pred_test)

train_precision = precision_score(y_train_binary, y_pred_train, zero_division=0)
test_precision = precision_score(y_test_binary, y_pred_test, zero_division=0)

train_recall = recall_score(y_train_binary, y_pred_train, zero_division=0)
test_recall = recall_score(y_test_binary, y_pred_test, zero_division=0)

train_f1 = f1_score(y_train_binary, y_pred_train, zero_division=0)
test_f1 = f1_score(y_test_binary, y_pred_test, zero_division=0)

# 打印性能结果
print("\nTraining Performance:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\nTest Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# 14. 绘制损失函数随迭代次数的变化
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(adaboost.losses) + 1), adaboost.losses, marker='o', label='Exponential Loss')
plt.xlabel('Number of Estimators (T)')
plt.ylabel('Loss')
plt.title('Exponential Loss vs. Number of Estimators')
plt.legend()
plt.grid()
plt.show()