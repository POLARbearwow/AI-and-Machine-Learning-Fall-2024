# main.py
import numpy as np
from data_preprocessing import load_and_preprocess_data
from evaluation import evaluate_model
from perception_model import PerceptronBGD, PerceptronSGD

def main():
    # 加载和预处理数据
    file_path = 'hw3/wine.data'  # 数据路径
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # 批量梯度下降感知机
    print("训练批量梯度下降感知机...")
    model_bgd = PerceptronBGD(n_features=X_train.shape[1], lr=0.1, n_iter=100)
    model_bgd.fit(X_train, y_train)
    y_pred_bgd = model_bgd.predict(X_test)
    evaluate_model(y_test, y_pred_bgd, "Batch Perceptron")
    model_bgd.plot_loss()  # 可视化损失变化

    # 随机梯度下降感知机
    print("训练随机梯度下降感知机...")
    model_sgd = PerceptronSGD(n_features=X_train.shape[1], lr=0.1, n_iter=100)
    model_sgd.fit(X_train, y_train)
    y_pred_sgd = model_sgd.predict(X_test)
    evaluate_model(y_test, y_pred_sgd, "Stochastic Perceptron")
    model_sgd.plot_loss()  # 可视化损失变化

if __name__ == '__main__':
    main()
