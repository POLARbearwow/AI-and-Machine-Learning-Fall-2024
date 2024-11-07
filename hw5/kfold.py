import numpy as np
from MLP import MLP

def k_fold_cross_validation(model_class, layer_sizes, x, y, k=5, epochs=10, batch_size=32, learning_rate=0.01):
    fold_size = x.shape[0] // k
    val_scores = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        x_val, y_val = x[start:end], y[start:end]
        x_train = np.concatenate([x[:start], x[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)
        
        model = model_class(layer_sizes)
        
        model.train(x_train, y_train, epochs, batch_size, learning_rate)
        
        predictions = model.forward(x_val)
        val_score = np.mean((predictions - y_val) ** 2)
        val_scores.append(val_score)
    
    return np.mean(val_scores)
