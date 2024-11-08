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
    [1, 10, 10, 1]  # 三个隐藏层，各有20个神经元
]

# 设置不同的 epochs 和 learning_rate
sgd_epochs = 200  # 初始设置的最大 epochs，提前停止会终止训练
sgd_learning_rate = 0.0001
mbgd_epochs = 200
mbgd_learning_rate = 0.0001

batch_sizes = [1, 16]
k_folds = 5

# 提前停止的参数
patience = 10  # 耐心次数
min_delta = 1e-2  # 最小改善值

# 创建一个大的画布，包含多个子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 的子图布局

print("Starting k-fold cross-validation for different configurations...")
index = 0
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
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            loss = mlp.train(x, y, epochs=1, batch_size=batch_size, learning_rate=learning_rate)  # 每次训练 1 个 epoch
            losses.append(loss[0])

            # 验证损失
            y_val_pred = mlp.forward(x)
            val_loss = np.mean((y - y_val_pred) ** 2)

            print(f"Epoch {epoch + 1}, Training Loss: {loss[0]:.4f}, Validation Loss: {val_loss:.4f}")

            # 提前停止逻辑
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0  # 重置耐心计数器
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.4f}")
                break

        # 获取当前的子图索引
        ax_loss = axs[index // 2, index % 2]

        # 绘制损失曲线在子图上
        ax_loss.plot(range(1, len(losses) + 1), losses, label=f'Layers {layer_sizes}, Batch size {batch_size} ({training_type})')
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Training Loss for layers {layer_sizes} and batch size {batch_size}")
        ax_loss.legend()

        # 获取下一个子图索引
        ax_pred = axs[(index + 1) // 2, (index + 1) % 2]

        # 生成预测值
        y_pred = mlp.forward(x)

        # 绘制真实数据和预测数据的对比图在子图上
        ax_pred.scatter(x, y, color='blue', label='Data with Noise')  # 真实数据
        ax_pred.plot(x, complex_function(x), color='red', label='True Function')  # 理想目标函数
        ax_pred.plot吧(x, y_pred, color='green', label='Model Prediction')  # 模型的预测
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("y")
        ax_pred.set_title(f"Model Prediction with layers {layer_sizes} and batch size {batch_size}")
        ax_pred.legend()

        # 增加索引
        index += 2

# 调整布局
plt.tight_layout()
plt.show()
print("\nTraining and validation complete.")
