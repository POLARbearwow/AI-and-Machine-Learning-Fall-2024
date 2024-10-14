import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def evaluate_model(y_true, y_pred, model_name):
    # 可选：如果 y_true 或 y_pred 中使用 -1 和 1 的标签，可以转换为 0 和 1
    y_pred = np.where(y_pred == -1, 0, y_pred)
    y_true = np.where(y_true == -1, 0, y_true)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{model_name} 的评估结果:")
    print(f"{'准确率:':<10} {accuracy:.4f}")
    print(f"{'召回率:':<10} {recall:.4f}")
    print(f"{'精确率:':<10} {precision:.4f}")
    print(f"{'F1 分数:':<10} {f1:.4f}")


# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# def evaluate_model(y_true, y_pred, model_name):
#     accuracy = accuracy_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred, average='binary')
#     precision = precision_score(y_true, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')

#     print(f"{model_name} 的评估结果:")
#     print(f"准确率: {accuracy:.4f}")
#     print(f"召回率: {recall:.4f}")
#     print(f"精确率: {precision:.4f}")
#     print(f"F1 分数: {f1:.4f}")
