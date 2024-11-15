import numpy as np

class MLP:
    # def __init__(self, layer_sizes):
    #     # 初始化权重和偏置
    #     self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    #     self.biases = [np.random.randn(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    
    # def __init__(self, layer_sizes):
    #         # Xavier 初始化
    #     self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1. / layer_sizes[i]) 
    #                     for i in range(len(layer_sizes) - 1)]
    #     self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
    
    def __init__(self, layer_sizes):
        # 使用 He 初始化
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]) 
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
    
    # def relu(self, x):
    #     # ReLU 激活函数
    #     return np.maximum(0, x)

    # def relu_derivative(self, x):
    #     # ReLU 函数的导数
    #     return np.where(x > 0, 1, 0)

    #leaky relu
    def relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


    # def forward(self, x):
    # # 前向传播
    #     self.layer_outputs = [x]
    #     for i, (W, b) in enumerate(zip(self.weights, self.biases)):
    #         x = self.relu(np.dot(x, W) + b)
    #         self.layer_outputs.append(x)
            
    #         # 打印当前层的输出值，检查是否为 0
    #         print(f"Layer {i + 1} output (first 5 values):", x[:5].flatten())  # 输出前 5 个值
    #     return x


    # def forward(self, x):
    #     # 前向传播
    #     self.layer_outputs = [x]
    #     for W, b in zip(self.weights, self.biases):
    #         x = self.relu(np.dot(x, W) + b)
    #         self.layer_outputs.append(x)
    #     return x

    def forward(self, x):
        # 前向传播
        self.layer_outputs = [x]
        for  W, b in zip(self.weights, self.biases):
            x = self.relu(np.dot(x, W) + b)
            self.layer_outputs.append(x)
            # 打印当前层的输出值，检查是否为 0
            #print(f" output (first 5 values):", x[:5].flatten())  # 输出前 5 个值
        return x

    def backward(self, y_true, learning_rate):
        # 反向传播
        delta = (self.layer_outputs[-1] - y_true) * self.relu_derivative(self.layer_outputs[-1])
        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(self.layer_outputs[i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.layer_outputs[i])

    # mse loss
    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        # 训练模型并记录每个 epoch 的损失
        n_samples = x_train.shape[0]
        losses = []
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]
            epoch_loss = 0
            
            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                y_pred = self.forward(x_batch)
                batch_loss = np.mean((y_pred - y_batch) ** 2)  # 计算 MSE
                epoch_loss += batch_loss
                self.backward(y_batch, learning_rate)
            
                #print(f"Epoch {epoch + 1}, Batch Loss: {batch_loss:.4f}")
                # print("y_pred (first 5 predictions):", y_pred[:5].flatten())
                # print("y_true (first 5 true values):", y_batch[:5].flatten())
            
            # 记录平均损失
            epoch_loss /= (n_samples // batch_size)
            losses.append(epoch_loss)
            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return losses
