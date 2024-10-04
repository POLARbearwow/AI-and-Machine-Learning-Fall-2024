import matplotlib.pyplot as plt

def plot_results(X, y, predictions, labels):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Original data')
    for norm_method, y_pred in predictions.items():
        ax.plot(X, y_pred, label=f'{norm_method} Normalization')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit with Different Normalization Methods')
    ax.legend()
    ax.grid(True)
    plt.show()
