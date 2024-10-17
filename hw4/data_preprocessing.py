import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # 定义列名 从.data文件中获取相关信息
    column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    df = pd.read_csv(file_path, names=column_names, sep=',')
    #print(df)
    
    df = df[df['Class'] != 3]  # 只保留两类数据 perception 只能处理二分类问题
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # 将标签 2 转换为 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    y_train = np.where(y_train == 2, -1, y_train)
    y_test = np.where(y_test == 2, -1, y_test)
    
    # print(df)
    # print('-------------------------------------------------------------------')
    # print(f"X_train shape: {X_train.shape}")
    # print(X_train)
    # print('-------------------------------------------------------------------')
    # print(f"X_test shape: {X_test.shape}")
    # print(X_test)
    
        # 打印标签的唯一值
    # print('-------------------------------------------------------------------')
    # print("y_train 的唯一值:", np.unique(y_train))
    # print("y_test 的唯一值:", np.unique(y_test))
    
    return X_train, X_test, y_train, y_test

load_and_preprocess_data('hw3/wine.data')