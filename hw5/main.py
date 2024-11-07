import numpy as np
from generate_dataset import generate_complex_regression_data, plot_data, complex_function
from MLP import MLP 
from kfold import k_fold_cross_validation  
import matplotlib.pyplot as plt
from evaluation import evaluate_classification

# 生成数据集
print("Generating complex regression data...")
x, y = generate_complex_regression_data(num_samples=200, noise=0.2)

# 设置不同的超参数组合，尝试找到最佳模型
layer_configs = [
    [1, 10, 10, 10, 1],  # 三个隐藏层，各有10个神经元
]

# 设置不同的 epochs 和 learning_rate
sgd_epochs = 50
sgd_learning_rate = 0.0001
mbgd_epochs = 100
mbgd_learning_rate = 0.00001

batch_sizes = [1, 16]
k_folds = 5

# 记录不同配置下的验证误差和预测效果
print("Starting k-fold cross-validation for different configurations...")
for layer_sizes in layer_configs:
    for batch_size in batch_sizes:
        # 根据 batch size 确定使用的 epochs 和 learning_rate
        if batch_size == 1:
            epochs = sgd_epochs
            learning_rate = sgd_learning_rate
            training_type = "SGD"
        else:
            epochs = mbgd_epochs
            learning_rate = mbgd_learning_rate
            training_type = "MBGD"

        print(f"\nTraining MLP with layer sizes: {layer_sizes}, batch size: {batch_size} ({training_type})")

        # 创建和训练模型
        mlp = MLP(layer_sizes)
        avg_val_score = k_fold_cross_validation(MLP, layer_sizes, x, y, k=k_folds, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        print(f"Average validation MSE for configuration {layer_sizes} with batch size {batch_size}: {avg_val_score:.4f}")

        # 训练模型并记录损失
        losses = mlp.train(x, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

        # 绘制损失曲线
        plt.figure()
        plt.plot(range(1, epochs + 1), losses, label=f'Layers {layer_sizes}, Batch size {batch_size} ({training_type})')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss for layers {layer_sizes} and batch size {batch_size}")
        plt.legend()

        # 生成预测值
        y_pred = mlp.forward(x)

        # 绘制真实数据和预测数据的对比图
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data with Noise')  # 真实数据
        plt.plot(x, complex_function(x), color='red', label='True Function')  # 理想目标函数
        plt.plot(x, y_pred, color='green', label='Model Prediction')  # 模型的预测
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Model Prediction with layers {layer_sizes} and batch size {batch_size}")
        plt.legend()

# 在最后一次调用 plt.show()，显示所有图
plt.show()
print("\nTraining and validation complete.")
