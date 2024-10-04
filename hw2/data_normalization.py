from linear_regression import LinearRegression

def normalize_data(X, y, norm_method):
    model = LinearRegression()
    if norm_method == 'Min-Max':
        X_normalized, X_min, X_max = model.min_max_normalize(X)
        y_normalized, y_min, y_max = model.min_max_normalize(y.reshape(-1, 1))
        y_normalized = y_normalized.flatten()
        return X_normalized, y_normalized, X_min, X_max, y_min, y_max
    elif norm_method == 'Mean':
        X_normalized, X_min, X_max = model.mean_normalize(X)
        y_normalized, y_mean, y_max = model.mean_normalize(y.reshape(-1, 1))
        y_normalized = y_normalized.flatten()
        return X_normalized, y_normalized, X_min, X_max, y_mean, y_max
    else:
        return X, y, None, None, None, None