import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionMiniBatch:
    def __init__(self, n_features, lr=0.1, n_iter=100, batch_size=32, tol=1e-4, patience=10):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.weights = np.random.randn(n_features + 1) * 0.05
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y_true, y_pred_prob):
        epsilon = 1e-10  # 防止 log(0)
        y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

    def _gradient(self, X, y_true, y_pred_prob):
        errors = y_pred_prob - y_true
        grad = np.dot(X.T, errors) / X.shape[0]
        return grad

    def _preprocess_data(self, X):
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])
        return X

    def _update_weights(self, X, y):
        no_improvement = 0
        for epoch in range(self.n_iter):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)  # 随机打乱数据
            X, y = X[indices], y[indices]
            
            epoch_loss = 0
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                
                y_pred_prob = self._sigmoid(np.dot(X_batch, self.weights))
                batch_loss = self.loss(y_batch, y_pred_prob)
                epoch_loss += batch_loss * len(y_batch)  # 累加批量损失

                grad = self._gradient(X_batch, y_batch, y_pred_prob)
                self.weights -= self.lr * grad  # 更新权重

            epoch_loss /= X.shape[0]  # 计算平均损失
            self.losses.append(epoch_loss)
            
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                no_improvement = 0

    def fit(self, X_train, y_train):
        X_train = self._preprocess_data(X_train)
        self._update_weights(X_train, y_train)

    def predict(self, X_test):
        X_test = self._preprocess_data(X_test)
        y_pred_prob = self._sigmoid(np.dot(X_test, self.weights))
        y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
        return y_pred
    
    
#------------------------------------SGD-------------------------------------

class LogisticRegressionSGD:
    def __init__(self, n_features, lr=0.1, n_iter=100, tol=1e-4, patience=10):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.patience = patience
        self.weights = np.random.randn(n_features + 1) * 0.05
        self.losses = []

    def _sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-z))

    def loss(self, y_true, y_pred_prob):
        """计算交叉熵损失"""
        epsilon = 1e-10  # 防止 log(0)
        y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

    def _gradient(self, x, y_true, y_pred_prob):
        """计算单样本的梯度"""
        error = y_pred_prob - y_true
        return x * error

    def _preprocess_data(self, X):
        """数据预处理：添加偏置项"""
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])
        return X

    def _update_weights(self, X, y):
        no_improvement = 0
        for epoch in range(self.n_iter):
            X_shuffled, y_shuffled = self._shuffle_data(X, y)  # 打乱数据
            epoch_loss = 0

            for i, x in enumerate(X_shuffled):
                y_pred_prob = self._sigmoid(np.dot(self.weights, x))
                loss = self.loss(y_shuffled[i], y_pred_prob)
                epoch_loss += loss
                grad = self._gradient(x, y_shuffled[i], y_pred_prob)  # 计算梯度
                self.weights -= self.lr * grad  # 更新权重

            avg_epoch_loss = epoch_loss / X.shape[0]
            self.losses.append(avg_epoch_loss)

            # tolerance
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                no_improvement = 0

    def _shuffle_data(self, X, y):
        """打乱数据"""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def fit(self, X_train, y_train):
        """训练模型"""
        X_train = self._preprocess_data(X_train)
        self._update_weights(X_train, y_train)

    def predict(self, X_test):
        """预测标签"""
        X_test = self._preprocess_data(X_test)
        y_pred_prob = self._sigmoid(np.dot(X_test, self.weights))
        y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
        return y_pred


#--------------------------------绘制损失曲线---------------------------------

def plot_losses(mbgd_loss, sgd_loss):
    plt.figure(figsize=(10, 6))
    
    # Mini-Batch Gradient Descent Loss Plot
    plt.subplot(2, 1, 1)
    plt.plot(mbgd_loss, label="Mini-Batch Gradient Descent Loss", color='blue')
    plt.title("Mini-Batch Gradient Descent Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Stochastic Gradient Descent Loss Plot
    plt.subplot(2, 1, 2)
    plt.plot(sgd_loss, label="Stochastic Gradient Descent Loss", color='orange')
    plt.title("Stochastic Gradient Descent Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
