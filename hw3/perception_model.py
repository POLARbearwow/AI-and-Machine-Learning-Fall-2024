import numpy as np
import matplotlib.pyplot as plt

class PerceptronBGD:
    def __init__(self, n_features, lr=0.1, n_iter=100, tol=0.001, patience=10):
        self.lr = lr  # 学习率
        self.n_iter = n_iter  # 最大迭代次数
        self.tol = tol  # 提前停止的容忍度
        self.patience = patience  # 提前停止的耐心
        self.weights = np.random.randn(n_features + 1) * 0.05  # 初始化权重 (包括偏置项)
        self.losses = []  # 记录每次迭代的损失

    def loss(self, y_true, y_pred):
        """计算损失 (误分类误差)"""
        errors = y_true * y_pred <= 0  # 找到误分类的样本
        return np.sum(-y_true[errors] * y_pred[errors])  # 只对误分类的样本计算损失
    
        #为什么不行
        #return np.sum(-y_true * y_pred[y_pred * y_true <= 0])  # 仅对误分类样本计算损失

    def _gradient(self, X, y_true, y_pred):
        """计算梯度 (基于整个数据集)"""
        errors = y_true * y_pred <= 0  # 只对错误分类的样本进行更新
        grad = np.dot(X[errors].T, y_true[errors])
        return grad

    def _preprocess_data(self, X):
        """数据预处理：添加偏置项"""
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])  # 增加一列偏置项
        return X

    def _update_weights(self, X, y):
        """使用批量梯度下降更新权重"""
        no_improvement = 0

        for epoch in range(self.n_iter):
            y_pred = np.sign(np.dot(X, self.weights))  # 全集的预测
            epoch_loss = self.loss(y, y_pred)  # 计算当前损失
            self.losses.append(epoch_loss)  # 记录损失

            # 计算整个训练集的梯度，并更新权重
            grad = self._gradient(X, y, y_pred)
            self.weights += self.lr * grad

            # 检查是否满足提前停止条件
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                no_improvement = 0  # 如果损失有改善，重置耐心值

    def fit(self, X_train, y_train):
        """训练模型"""
        X_train = self._preprocess_data(X_train)
        self._update_weights(X_train, y_train)

    def predict(self, X_test):
        """预测标签"""
        X_test = self._preprocess_data(X_test)
        y_pred = np.sign(np.dot(X_test, self.weights))
        return y_pred

    # def plot_loss(self):
    #     """绘制损失随迭代次数的变化"""
    #     plt.plot(self.losses)
    #     plt.title("Loss over epochs (Batch Gradient Descent)")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.show()

#------------------------------------SGD-------------------------------------

class PerceptronSGD:
    def __init__(self, n_features, lr=1, n_iter=100, tol=0.00001, patience=10):
        print('-------------------------------- Learning rate:', lr)
        self.lr = lr  # Learning rate
        self.n_iter = n_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for early stopping
        self.patience = patience  # Patience for early stopping
        self.weights = np.random.randn(n_features + 1) * 0.05  # Model parameters (include bias term)
        self.losses = []  # Track loss over time

    def loss(self, y_true, y_pred):
        """Compute the loss (misclassification error)."""
        return max(0, -y_true * y_pred)  # Ensure loss is non-negative
    
    def _gradient(self, x, y_true, y_pred):
        """Compute the gradient for updating weights."""
        return y_true * x if y_true * y_pred <= 0 else 0

    def _preprocess_data(self, X):
        """Add bias term to the input data."""
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])  # Add bias as first column
        return X

    def _update_weights(self, X, y):
        """Update the weights using stochastic gradient descent."""
        no_improvement = 0  # Move initialization outside loop for global tracking
        for epoch in range(self.n_iter):
            X_shuffled, y_shuffled = self._shuffle_data(X, y)  # Shuffle the data
            epoch_loss = 0

            for i, x in enumerate(X_shuffled):
                y_pred = np.sign(np.dot(self.weights, x))
                loss = self.loss(y_shuffled[i], y_pred)
                epoch_loss += loss
                if loss > 0:
                    # Update weights
                    grad = self._gradient(x, y_shuffled[i], y_pred)
                    self.weights += self.lr * grad
 
            # Record and check loss for early stopping
            self.losses.append(epoch_loss)
            #print(f'Epoch {epoch+1}, Loss: {epoch_loss}')  # Print loss for debugging

            if epoch_loss < self.tol:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping triggered after {no_improvement} epochs.")
                    break
            else:
                no_improvement = 0  # Reset no_improvement if the loss decreases

    def _shuffle_data(self, X, y):
        """Shuffle the dataset."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def fit(self, X_train, y_train):
        """Train the model using SGD."""
        X_train = self._preprocess_data(X_train)
        self.weights = np.random.randn(X_train.shape[1]) * 0.05  # Re-initialize weights for each fit
        self._update_weights(X_train, y_train)

    def predict(self, X_test):
        """Predict the labels for test data."""
        X_test = self._preprocess_data(X_test)
        y_pred = np.sign(np.dot(X_test, self.weights))
        return y_pred

    # def plot_loss(self):
    #     """Plot the loss over time."""
    #     plt.plot(self.losses)
    #     plt.title("Loss over iterations")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.show()

def plot_losses(bgd_loss, sgd_loss):
    plt.figure(figsize=(10, 6))  # 设置画布大小

    # 第一幅图：Batch Gradient Descent Loss
    plt.subplot(2, 1, 1)  # 2行1列，第一幅图
    plt.plot(bgd_loss, label="Batch Gradient Descent Loss", color='blue')
    plt.title("Batch Gradient Descent Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 第二幅图：Stochastic Gradient Descent Loss
    plt.subplot(2, 1, 2)  # 2行1列，第二幅图
    plt.plot(sgd_loss, label="Stochastic Gradient Descent Loss", color='orange')
    plt.title("Stochastic Gradient Descent Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 调整布局，避免重叠
    plt.tight_layout()  
    plt.show()

# from sklearn.linear_model import Perceptron

# def train_batch_perceptron(X_train, y_train):
#     # 定义批量更新感知机
#     batch_perceptron = Perceptron(max_iter=1000, tol=1e-3)
#     batch_perceptron.fit(X_train, y_train)
#     return batch_perceptron

# def train_stochastic_perceptron(X_train, y_train):
#     # 定义随机更新感知机
#     stochastic_perceptron = Perceptron(max_iter=1, tol=None, shuffle=True, random_state=42)
#     stochastic_perceptron.fit(X_train, y_train)
#     return stochastic_perceptron

# from sklearn.linear_model import SGDClassifier

# # 批量更新感知机
# def train_batch_perceptron(X_train, y_train):
#     # 设置较大的 batch_size 实现近似批量更新
#     batch_perceptron = SGDClassifier(loss='perceptron', max_iter=1000, tol=1e-3, learning_rate='constant', eta0=1.0)
#     batch_perceptron.fit(X_train, y_train)
#     return batch_perceptron

# # 随机更新感知机
# def train_stochastic_perceptron(X_train, y_train):
#     # 逐个样本进行更新 (随机梯度下降)
#     stochastic_perceptron = SGDClassifier(loss='perceptron', max_iter=1, tol=None, shuffle=True, random_state=42)
#     stochastic_perceptron.fit(X_train, y_train)
#     return stochastic_perceptron
