from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(y_true, y_pred):
    """
    计算并打印分类模型的各种评估指标，包括准确率、精确率、召回率和 F1 分数。
    
    参数:
    - y_true (np.ndarray): 真实的标签
    - y_pred (np.ndarray): 模型预测的标签
    
    返回:
    - metrics (dict): 包含所有计算指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # 输出指标
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 返回指标字典
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics
