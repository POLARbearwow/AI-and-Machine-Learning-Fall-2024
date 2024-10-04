import numpy as np  
import matplotlib.pyplot as plt
from generate_dataset import generate_linear_data, add_bias
from plotting import plot_results
from linear_regression import LinearRegression
from data_normalization import normalize_data



def main():
    X_train, y_train = generate_linear_data(1, 10 ,100)
    # 增加一列偏置项
    X_train_b = add_bias(X_train)

    # 初始化归一化方法（按顺序先 Min-Max 再 None 处理）
    normalization_methods = {
        'Min-Max': LinearRegression().min_max_normalize(X_train_b),
        'Mean': LinearRegression().mean_normalize(X_train_b),
        'None': (X_train_b, y_train)  # 注意，这里没有 X_min 和 X_max
    }

    all_fitted_lines = {}

    for norm_method, values in normalization_methods.items():
        print(f"\nTesting with {norm_method} normalization...")
        if norm_method == 'None':
            # 'None' 归一化不需要 X_min 和 X_max
            X_normalized, y_normalized = values
            model = LinearRegression(learning_rate=0.000001, epochs=1000)
        else:
            X_normalized, X_min, X_max = values
            # 对 y_train 也进行归一化处理（如果是 Min-Max 或 Mean 归一化）
            if norm_method == 'Min-Max':
                y_normalized, y_min, y_max = LinearRegression().min_max_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()
                model = LinearRegression(learning_rate=0.1, epochs=1000)

            elif norm_method == 'Mean':
                y_normalized, y_mean, y_max = LinearRegression().mean_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()
                model = LinearRegression(learning_rate=0.1, epochs=1000)


        # 使用批量梯度下降（BGD）进行训练
        model.bgd(X_normalized, y_normalized)
        
        # 保存每种归一化方法的预测结果
        y_pred = model.predict(X_normalized)

        # 如果归一化了 y_train，则进行反归一化处理
        if norm_method == 'Min-Max':
            y_pred = y_pred * (y_max - y_min) + y_min
        elif norm_method == 'Mean':
            y_pred = y_pred * (y_max - y_min) + y_mean

        all_fitted_lines[norm_method] = y_pred     
           
        #print(f"-------{all_fitted_lines}-----")
        labels = []
        
    plot_results(X_train, y_train, all_fitted_lines, labels)
    
if __name__ == "__main__":
       main()