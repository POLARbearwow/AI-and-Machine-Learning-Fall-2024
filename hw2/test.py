#有点问题

# import numpy as np  
# import matplotlib.pyplot as plt

# class LinearRegression:
#     def __init__(self, learning_rate=0.0001, epochs=1000):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.weights = None
#         self.losses = []
    
#     def mean_squared_error(self, true, pred):
#         squared_error = np.square(true - pred)
#         mse_loss = np.mean(squared_error)
#         return mse_loss
    
#     def min_max_normalize(self, X):
#         X_min = np.min(X, axis=0)  # 列操作
#         X_max = np.max(X, axis=0)
#         diff = X_max - X_min
#         diff[diff == 0] = 1  # 防止除以0
#         return (X - X_min) / diff, X_min, X_max

#     def mean_normalize(self, X):
#         X_mean = np.mean(X, axis=0)
#         X_min = np.min(X, axis=0)
#         X_max = np.max(X, axis=0)
#         diff = X_max - X_min
#         diff[diff == 0] = 1  # 防止除以0
#         return (X - X_mean) / diff, X_mean, X_max
    
#     def sgd(self, X, y):
#         m, n = X.shape
#         self.weights = np.zeros(n)
#         self.losses = []
        
#         for epoch in range(self.epochs):
#             for i in range(m):
#                 x_i = X[i]
#                 y_i = y[i]
#                 y_pred = np.dot(x_i, self.weights)
                
#                 gradient = -x_i * (y_i - y_pred)
#                 self.weights -= self.learning_rate * gradient
            
#             loss = self.mean_squared_error(y, np.dot(X, self.weights))
#             self.losses.append(loss)
#             if epoch % 100 == 0:
#                 print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
    
#     def bgd(self, X, y):
#         m, n = X.shape
#         self.weights = np.zeros(n)
#         self.losses = []
        
#         for epoch in range(self.epochs):
#             y_pred = np.dot(X, self.weights)
#             gradient = -np.dot(X.T, (y - y_pred)) / m
#             self.weights -= self.learning_rate * gradient
            
#             loss = self.mean_squared_error(y, y_pred)
#             self.losses.append(loss)
#             if epoch % 100 == 0:
#                 print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
    
#     def mbgd(self, X, y, batch_size):
#         m, n = X.shape
#         self.weights = np.zeros(n)
#         self.losses = []
        
#         for epoch in range(self.epochs):
#             for i in range(0, m, batch_size):
#                 x_i = X[i:i+batch_size]
#                 y_i = y[i:i+batch_size]
#                 y_pred = np.dot(x_i, self.weights)
                
#                 gradient = -np.dot(x_i.T, (y_i - y_pred)) / batch_size
#                 self.weights -= self.learning_rate * gradient
            
#             loss = self.mean_squared_error(y, np.dot(X, self.weights))
#             self.losses.append(loss)
#             if epoch % 100 == 0:
#                 print(f'Epoch {epoch}: Weights {self.weights}, Loss {loss}')
    
#     def predict(self, X):
#         return np.dot(X, self.weights)

#     def reverse_weights(self, X_min, X_max):
#         # 如果进行了归一化，则需要对权重进行反归一化处理
#         w_0 = self.weights[0]  # 截距项
#         w_1_to_n = self.weights[1:]  # 斜率项
#         w_1_to_n_original = w_1_to_n / (X_max[1:] - X_min[1:])  # 反归一化斜率
#         w_0_original = w_0 - np.sum(w_1_to_n_original * X_min[1:])  # 反归一化截距
#         return np.concatenate([[w_0_original], w_1_to_n_original])  # 返回反归一化后的权重


# if __name__ == "__main__":
#     # 使用噪声数据集
#     X_train = np.arange(100).reshape(100, 1)
#     a, b = 1, 10
#     y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
#     y_train = y_train.reshape(-1)  # 将y展平成一维数组

#     # 增加一列偏置项
#     X_train_b = np.c_[np.ones((100, 1)), X_train]

#     # 初始化归一化方法（按顺序先 Min-Max 再 None 处理）
#     normalization_methods = {
#         'Min-Max': LinearRegression().min_max_normalize(X_train_b),
#         'Mean': LinearRegression().mean_normalize(X_train_b),
#         'None': (X_train_b, y_train)  # 注意，这里没有 X_min 和 X_max
#     }

#     all_fitted_lines = {}

#     # 测试不同的归一化方法
#     for norm_method, values in normalization_methods.items():
#         print(f"\nTesting with {norm_method} normalization...")

#         if norm_method == 'None':
#             # 'None' 归一化不需要 X_min 和 X_max
#             X_normalized, y_normalized = values
#             X_min, X_max = None, None
#             model = LinearRegression(learning_rate=0.00001, epochs=1000)  # 使用较小的学习率
#         else:
#             X_normalized, X_min, X_max = values
#             # 对 y_train 也进行归一化处理（如果是 Min-Max 或 Mean 归一化）
#             if norm_method == 'Min-Max':
#                 y_normalized, y_min, y_max = LinearRegression().min_max_normalize(y_train.reshape(-1, 1))
#                 y_normalized = y_normalized.flatten()  # 保证 y_normalized 是一维的
#             elif norm_method == 'Mean':
#                 y_normalized, y_mean, y_max = LinearRegression().mean_normalize(y_train.reshape(-1, 1))
#                 y_normalized = y_normalized.flatten()  # 保证 y_normalized 是一维的

#             model = LinearRegression(learning_rate=0.1, epochs=1000)

#         # 使用批量梯度下降（BGD）进行训练
#         model.bgd(X_normalized, y_normalized)
        
#         # 保存每种归一化方法的预测结果
#         y_pred = model.predict(X_normalized)

#         # 如果归一化了 y_train，则进行反归一化处理
#         if norm_method == 'Min-Max':
#             y_pred = y_pred * (y_max - y_min) + y_min
#         elif norm_method == 'Mean':
#             y_pred = y_pred * (y_max - y_min) + y_mean

#         # 对权重进行反归一化
#         if norm_method != 'None':
#             final_weights_original = model.reverse_weights(X_min, X_max)
#             print(f"Final weights (original scale) for {norm_method}: {final_weights_original}")
            
#             # 根据反归一化的权重绘制拟合曲线
#             slope = final_weights_original[1]
#             intercept = final_weights_original[0]
#             y_fitted = slope * X_train + intercept
#         else:
#             print(f"Final weights (original scale) for {norm_method}: {model.weights}")
#             slope = model.weights[1]
#             intercept = model.weights[0]
#             y_fitted = slope * X_train + intercept

#         all_fitted_lines[norm_method] = y_fitted

#     # 绘制不同归一化方法的拟合结果在一张图中
#     fig, ax = plt.subplots()
#     ax.scatter(X_train, y_train, color='blue', label='Original data')
    
#     # 依次绘制每种归一化方法的结果，包括 None 归一化
#     for norm_method, y_fitted in all_fitted_lines.items():
#         ax.plot(X_train, y_fitted, label=f'{norm_method} Normalization')

#     ax.set_xlabel('X')
#     ax.set_ylabel('y')
#     ax.set_title('Linear Regression Fit with Reversed Weights')
#     ax.legend()
#     ax.grid(True)

#     # 显示所有图
#     plt.show()
