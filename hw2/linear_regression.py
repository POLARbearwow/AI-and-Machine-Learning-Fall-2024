import numpy as np

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
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
    
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
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
    
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
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
                
    def predict(self, X):
        print(f'Final weights: {self.weights}')
        return np.dot(X, self.weights)