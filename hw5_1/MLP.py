import numpy as np
import matplotlib.pyplot as plt

class Module:
    def __init__(self, lr=None):
        self.linear1 = Linear(1, 15, batch_size=200)  # 输入特征是1, 批量大小设置为200
        self.linear1.first_layer = True
        self.linear1.add_Motivation('sigmoid')
        self.linear2 = Linear(15, 5, batch_size=200)
        self.linear2.add_Motivation('relu')
        self.linear3 = Linear(5, 1, batch_size=200)
        self.linear3.last_layer = True
        self.lr = 1e-3  # 学习率
        if lr is not None:
            self.lr = lr

    def forward(self, x):
        x = x.copy()
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        x = self.linear3.forward(x)
        return x

    def compute_gradient(self, x):
        x = x.copy()
        self.linear3.compute_local_gradient(x)
        x = self.linear3.to_last.copy()
        self.linear2.compute_local_gradient(x)
        x = self.linear2.to_last.copy()
        self.linear1.compute_local_gradient(x)

    def backward(self):
        self.linear3.backward(self.lr)
        self.linear2.backward(self.lr)
        self.linear1.backward(self.lr)


class Linear:
    def __init__(self, in_feature, out_feature, batch_size=None, bias=None):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Oin = None
        self.Oout = None
        self.Onet = None
        self.bias = True
        self.batch_size = 1
        if batch_size is not None:
            self.batch_size = batch_size
        if bias:
            self.bias = bias
        self.Weights = np.random.randn(self.out_feature, self.in_feature + 1)
        self.local_gradient = None
        self.motivation = None
        self.last_layer = False
        self.to_last = None
        self.first_layer = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.Oin = x.copy()
        ones = np.ones((1, self.batch_size))  # 在这里，ones的shape应该是 (1, batch_size)
        self.Oin = np.row_stack((self.Oin, ones))  # 将ones行堆叠到输入矩阵的最后一行
        self.Onet = self.Weights.dot(self.Oin)
        if self.motivation is None:
            self.Oout = self.Onet.copy()
        elif self.motivation == 'sigmoid':
            self.Oout = self.sigmoid(self.Onet)
        elif self.motivation == 'relu':
            self.Oout = self.relu(self.Onet)
        return self.Oout

    def add_Motivation(self, TT):
        if TT == 'sigmoid':
            self.motivation = 'sigmoid'
        elif TT == 'relu':
            self.motivation = 'relu'

    def compute_local_gradient(self, x):
        if self.last_layer:
            tp = -(x - self.Oout)
            self.local_gradient = tp * self.motivation_back
        else:
            self.local_gradient = self.motivation_back * x
        to_l = []
        for i in range(self.batch_size):
            a = []
            for j in range(self.in_feature):
                a.append(self.Weights[:, j] * self.local_gradient[:, i])
            to_l.append(np.sum(a, axis=1))
        to_l = np.array(to_l).T
        self.to_last = to_l.copy()

    def backward(self, lr):
        g = self.local_gradient.dot(self.Oin.T / self.batch_size)
        self.Weights = self.Weights - g * lr

    @property
    def motivation_back(self):
        if self.motivation == 'sigmoid':
            return self.sigmoid(self.Onet) * (1 - self.sigmoid(self.Onet))
        elif self.motivation == 'relu':
            tp = self.Onet.copy()
            tp[tp > 0] = 1
            tp[tp <= 0] = 0
            return tp
        else:
            return np.ones(np.shape(self.Onet))


def main():
    # 生成训练数据
    x_train = np.linspace(0, 2 * np.pi, 200).reshape(200, 1)  # 输入是从0到2π的200个点
    y_train = np.sin(x_train)  # 目标输出是sin(x)

    # 给y_train添加噪声，噪声为正态分布，均值为0，标准差为0.1
    noise = np.random.normal(0, 0.1, y_train.shape)  # 添加噪声
    y_train_noisy = y_train + noise  # 带噪声的目标输出

    # 初始化网络
    model = Module(lr=1e-1)

    # 训练过程
    epochs = 5000
    for epoch in range(epochs):
        # 前向传播
        y_pred = model.forward(x_train.T)

        # 计算梯度并反向传播
        model.compute_gradient(y_train_noisy.T)  # 使用带噪声的数据训练
        model.backward()

        # 每1000次打印一次损失
        if epoch % 1000 == 0:
            loss = np.mean((y_train_noisy.T - y_pred) ** 2)  # 计算带噪声的数据的损失
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss:.4f}")

    # 测试模型
    y_test = model.forward(x_train.T)

    # 绘制结果
    plt.plot(x_train, y_train, label="True Function (sin(x))")  # 绘制真实函数
    plt.plot(x_train, y_train_noisy, label="Noisy Function")  # 绘制带噪声的数据
    plt.plot(x_train, y_test.T, label="Predicted Function")  # 绘制预测结果
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
