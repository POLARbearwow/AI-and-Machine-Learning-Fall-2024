import matplotlib.pyplot as plt

# def plot_results(X, y, predictions, labels, rate1, rate2):
#     fig, ax = plt.subplots()
#     ax.scatter(X, y, color='blue', label='Original data')
#     for norm_method, y_pred in predictions.items():
#         ax.plot(X, y_pred, label=f'{norm_method} Normalization')
#     ax.set_xlabel('X')
#     ax.set_ylabel('y')
#     ax.set_title('Linear Regression Fit with Different Normalization Methods')
#     ax.legend()
#     ax.grid(True)
#     plt.show()

import matplotlib.pyplot as plt

def plot_results(X, y, predictions, labels, rate1, rate2):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Original data')
    
    # 迭代 predictions 字典，这里假设 keys 是归一化方法的名称
    for norm_method, y_pred in predictions.items():
        if norm_method == 'None':
            label = f'{norm_method} Normalization LR: {rate1}'  # 为 'None' 方法添加 rate1
        else:
            label = f'{norm_method} Normalization LR: {rate2}'  # 为 'Min-Max' 或 'Mean' 添加 rate2
        ax.plot(X, y_pred, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit with Different Normalization Methods')
    ax.legend()
    ax.grid(True)
    plt.show()

