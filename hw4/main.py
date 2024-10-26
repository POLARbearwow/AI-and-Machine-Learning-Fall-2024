# main.py
import numpy as np
from data_preprocessing import load_and_preprocess_data
from evaluation import evaluate_model
from logistic_regression import LogisticRegressionMiniBatch, LogisticRegressionSGD, plot_losses


def main():
    # Load and preprocess data
    file_path = 'hw4/wine.data'  # Data path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Mini-Batch Gradient Descent Logistic Regression
    print("Training logistic regression with mini-batch gradient descent...")
    model_mbgd = LogisticRegressionMiniBatch(n_features=X_train.shape[1], lr=1e-5, n_iter=5000,tol=0.01)
    model_mbgd.fit(X_train, y_train)
    y_pred_bgd = model_mbgd.predict(X_test)
    evaluate_model(y_test, y_pred_bgd, "Mini-Batch Logistic Regression")
    
    
    # Stochastic Gradient Descent Logistic Regression
    print("Training logistic regression with stochastic gradient descent...")
    model_sgd = LogisticRegressionSGD(n_features=X_train.shape[1], lr=5e-6, n_iter=5000,tol=0.5)
    model_sgd.fit(X_train, y_train)
    y_pred_sgd = model_sgd.predict(X_test)
    evaluate_model(y_test, y_pred_sgd, "Stochastic Logistic Regression")
    
    plot_losses(model_mbgd.losses, model_sgd.losses)
    
    # print("Mini-Batch Losses:", model_mbgd.losses)
    # print("Stochastic Losses:", model_sgd.losses)
    

if __name__ == '__main__':
    main()
