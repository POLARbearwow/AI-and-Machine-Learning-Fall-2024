import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation_func='relu', init_method='he'):
        """
        初始化MLP网络。
        layer_sizes: 包含各层神经元数量的列表，形式为 [输入层, 隐藏层1, 隐藏层2, ..., 输出层]
        activation_func: 默认激活函数为 'relu'，可选 'sigmoid'
        init_method: 权重初始化方法，可选 'he'（He初始化），'xavier'（Xavier初始化）和 'random'（标准正态初始化）
        """
        self.activation_func = activation_func
        self.weights, self.biases = self.initialize_weights(layer_sizes, init_method)

    def initialize_weights(self, layer_sizes, init_method):
        """
        根据指定的初始化方法初始化权重和偏置。
        init_method: 权重初始化方法，可选 'he', 'xavier', 'random'
        """
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            if init_method == 'he':
                # He初始化 (适用于ReLU激活函数)
                weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            elif init_method == 'xavier':
                # Xavier初始化 (适用于Sigmoid和Tanh激活函数)
                weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1. / layer_sizes[i])
            else:
                # 默认标准正态分布初始化
                weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])

            bias = np.zeros((1, layer_sizes[i + 1]))

            weights.append(weight)
            biases.append(bias)
        
        return weights, biases
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
        
    # def sigmoid(self, z):
    #     return 1 / (1 + np.exp(-z))

    # def sigmoid_derivative(self, z):
    #     sig = self.sigmoid(z)
    #     return sig * (1 - sig)

    def forward(self, x):
        self.layer_outputs = [x]
        for i in range(len(self.weights)):
            weight = self.weights[i]
            bias = self.biases[i]

            z = np.dot(self.layer_outputs[-1], weight) + bias

            if i < len(self.weights) - 1:  
                if self.activation_func == 'relu':
                    a = self.relu(z)
                elif self.activation_func == 'sigmoid':
                    a = self.sigmoid(z)
            else:  
                a = z

            self.layer_outputs.append(a)

        return self.layer_outputs[-1]


    def backward(self, x, y, learning_rate):
        m = y.shape[0]
     
        delta_o = self.layer_outputs[-1] - y 

        for i in reversed(range(len(self.weights))):

            grad_w = np.dot(self.layer_outputs[i].T, delta_o) / m
            grad_b = np.sum(delta_o, axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

            if i != 0:
                if self.activation_func == 'relu':
                    delta_z = delta_o.dot(self.weights[i].T) * self.relu_derivative(self.layer_outputs[i])
                elif self.activation_func == 'sigmoid':
                    delta_z = delta_o.dot(self.weights[i].T) * self.sigmoid_derivative(self.layer_outputs[i])
                delta_o = delta_z


    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        n_samples = x_train.shape[0]
        losses = []
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            epoch_loss = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                y_pred = self.forward(x_batch)
                
                batch_loss = np.mean((y_batch - y_pred) ** 2)
                epoch_loss += batch_loss

                self.backward(x_batch, y_batch, learning_rate)

            epoch_loss /= (n_samples // batch_size)
            losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        return losses
