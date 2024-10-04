def inverse_normalize(y_pred, y_min, y_max, method='Min-Max'):
    if method == 'Min-Max' or method == 'Mean':
        return y_pred * (y_max - y_min) + y_min
    return y_pred
