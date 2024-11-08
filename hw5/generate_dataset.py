import numpy as np
import matplotlib
#matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
import threading


def complex_function(x):
    # 定义一个复杂的非线性函数 y = sin(x) + cos(x) + sin(2x)
    return np.sin(x) + np.cos(x) + np.sin(2 * x) + 5
    #return np.sin(x) + 5 

def generate_complex_regression_data(num_samples=500, noise=0.1):
    # 生成输入数据 x
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(-1, 1)
    # 计算目标 y 值并添加噪声
    y = complex_function(x) + noise * np.random.randn(num_samples, 1)

    return x, y

def plot_data(x, y):
    # 启用交互模式
    #plt.ion()
    plt.scatter(x, y, color='blue', label='Data with Noise')
    plt.plot(x, complex_function(x), color='red', label='True Function')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Complex Nonlinear Function Data")
    plt.legend()
    plt.show()
