import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # 定义列名 从.data文件中获取相关信息
    column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    df = pd.read_csv(file_path, names=column_names, sep=',')
    print(df)
    
    df = df[df['Class'] != 3]  # 只保留两类数据 perception 只能处理二分类问题
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(df)
    print("训练集特征数据:", X_train)
    print("测试集特征数据:", X_test)
    print("训练集标签数据:", y_train)
    print("测试集标签数据:", y_test)
    
load_and_preprocess_data('hw3/wine.data')