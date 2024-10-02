import numpy as np  
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.losses = []
    
    def mean_squared_error(self, true, pred):
        squared_error = np.square(true - pred)
        mse_loss = np.mean(squared_error)
        return mse_loss
    
    def min_max_normalize(self, X):
        X_min = np.min(X, axis=0)  # 列操作
        X_max = np.max(X, axis=0)
        diff = X_max - X_min
        diff[diff == 0] = 1  # 防止除以0
        return (X - X_min) / diff, X_min, X_max

    def mean_normalize(self, X):
        X_mean = np.mean(X, axis=0)
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        diff = X_max - X_min
        diff[diff == 0] = 1  # 防止除以0
        return (X - X_mean) / diff, X_mean, X_max
    
    def sgd(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.losses = []
        
        for epoch in range(self.epochs):
            for i in range(m):
                x_i = X[i]
                y_i = y[i]
                y_pred = np.dot(x_i, self.weights)
                
                gradient = -x_i * (y_i - y_pred)
                self.weights -= self.learning_rate * gradient
            
            loss = self.mean_squared_error(y, np.dot(X, self.weights))
            self.losses.append(loss)
    
    def bgd(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.losses = []
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights)
            gradient = -np.dot(X.T, (y - y_pred)) / m
            self.weights -= self.learning_rate * gradient
            
            loss = self.mean_squared_error(y, y_pred)
            self.losses.append(loss)
    
    def mbgd(self, X, y, batch_size):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.losses = []
        
        for epoch in range(self.epochs):
            for i in range(0, m, batch_size):
                x_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]
                y_pred = np.dot(x_i, self.weights)
                
                gradient = -np.dot(x_i.T, (y_i - y_pred)) / batch_size
                self.weights -= self.learning_rate * gradient
            
            loss = self.mean_squared_error(y, np.dot(X, self.weights))
            self.losses.append(loss)
    
    def predict(self, X):
        print(f'111{self.weights}')
        return np.dot(X, self.weights)


if __name__ == "__main__":
    # 使用噪声数据集
    X_train = np.arange(100).reshape(100, 1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1)  # 将y展平成一维数组

    # 增加一列偏置项
    X_train_b = np.c_[np.ones((100, 1)), X_train]

    # 初始化归一化方法（按顺序先 Min-Max 再 None 处理）
    normalization_methods = {
        'Min-Max': LinearRegression().min_max_normalize(X_train_b),
        'Mean': LinearRegression().mean_normalize(X_train_b),
        'None': (X_train_b, y_train)  # 注意，这里没有 X_min 和 X_max
    }

    all_fitted_lines = {}

    for norm_method, values in normalization_methods.items():
        print(f"\nTesting with {norm_method} normalization...")

        if norm_method == 'None':
            # 'None' 归一化不需要 X_min 和 X_max
            X_normalized, y_normalized = values
        else:
            X_normalized, X_min, X_max = values
            # 对 y_train 也进行归一化处理（如果是 Min-Max 或 Mean 归一化）
            if norm_method == 'Min-Max':
                y_normalized, y_min, y_max = LinearRegression().min_max_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()
            elif norm_method == 'Mean':
                y_normalized, y_mean, y_max = LinearRegression().mean_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()

        model = LinearRegression(learning_rate=0.1, epochs=1000)

        # 使用批量梯度下降（BGD）进行训练
        model.bgd(X_normalized, y_normalized)
        
        # 保存每种归一化方法的预测结果
        y_pred = model.predict(X_normalized)

        # 如果归一化了 y_train，则进行反归一化处理
        if norm_method == 'Min-Max':
            y_pred = y_pred * (y_max - y_min) + y_min
        elif norm_method == 'Mean':
            y_pred = y_pred * (y_max - y_min) + y_mean

        all_fitted_lines[norm_method] = y_pred

    # 绘制不同归一化方法的拟合结果在一张图中
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color='blue', label='Original data')
    
    # 依次绘制每种归一化方法的结果，包括 None 归一化
    for norm_method, y_pred in all_fitted_lines.items():
        ax.plot(X_train, y_pred, label=f'{norm_method} Normalization')

    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit with Different Normalization Methods')
    ax.legend()
    ax.grid(True)

    # 显示所有图
    plt.show()
