import numpy as np  
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass
    
    def mean_squared_error(self,true,pred):
        squraed_error = np.square(true - pred)
        sum_squared_error = np.sum(squraed_error)
        mse_loss = sum_squared_error/ true.size
        return mse_loss