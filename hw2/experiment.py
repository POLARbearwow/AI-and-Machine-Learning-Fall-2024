import numpy as np  
import matplotlib.pyplot as plt
from generate_dataset import generate_linear_data, add_bias
from plotting import plot_results
from linear_regression import LinearRegression
from data_normalization import normalize_data
from plot_loss import plot_losses



def main():
    X_train, y_train = generate_linear_data(1, 10 ,100)
    X_train_b = add_bias(X_train)

    normalization_methods = {
        'Min-Max': LinearRegression().min_max_normalize(X_train_b),
        'Mean': LinearRegression().mean_normalize(X_train_b),
        'None': (X_train_b, y_train)  
    }
    
    gradient_methods = {
        'BGD': 'bgd',
        'SGD': 'sgd',
        'MBGD': 'mbgd'
    }

    all_fitted_lines = {}
    all_losses = {}
    
    #for gradient_method, method in gradient_methods.items():
    learning_rate1=0.0001 ;learning_rate2=0.1 ; epochs=1000
    for norm_method, values in normalization_methods.items():
        print(f"\nTesting with {norm_method} normalization...")
        if norm_method == 'None':
            X_normalized, y_normalized = values
            model = LinearRegression(learning_rate1, epochs)
        else:
            X_normalized, X_min, X_max = values
            if norm_method == 'Min-Max':
                y_normalized, y_min, y_max = LinearRegression().min_max_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()
                model = LinearRegression(learning_rate2, epochs)

            elif norm_method == 'Mean':
                y_normalized, y_mean, y_max = LinearRegression().mean_normalize(y_train.reshape(-1, 1))
                y_normalized = y_normalized.flatten()
                model = LinearRegression(learning_rate2, epochs)
        
        #model.mbgd(X_normalized, y_normalized, batch_size=10)
        #model.sgd(X_normalized, y_normalized)

        all_losses[norm_method] = model.losses
        
        y_pred = model.predict(X_normalized)

        # 进行反归一化处理
        if norm_method == 'Min-Max':
            y_pred = y_pred * (y_max - y_min) + y_min
        elif norm_method == 'Mean':
            y_pred = y_pred * (y_max - y_min) + y_mean

        all_fitted_lines[norm_method] = y_pred     
        
        #print(f"-------{all_fitted_lines}-----")
        labels = []
        
    plot_results(X_train, y_train, all_fitted_lines, labels, learning_rate1, learning_rate2)
    
    # print("plot start-------------------------")
    plot_losses(all_losses)
    # for norm_method, losses in all_losses.items():
    #     print(f"{norm_method}: Loss over epochs: {losses[:10]}...")  # 只打印每种方法的前10个损失值以简化输出
    # print("plot end---------------------------")
    
if __name__ == "__main__":
       main()