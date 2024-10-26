import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    df = pd.read_csv(file_path, names=column_names, sep=',')
    #print(df)
    
    df = df[df['Class'] != 3]  
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    y_train = np.where(y_train == 2, 0, y_train)
    y_test = np.where(y_test == 2, 0, y_test)

    return X_train, X_test, y_train, y_test

load_and_preprocess_data('hw4/wine.data')