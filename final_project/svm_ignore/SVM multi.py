import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        for _ in range(self.max_iters):
            loss = 0
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    loss += 1 - y[idx] * (np.dot(x_i, self.w) + self.b)

            loss += self.lambda_param * np.sum(self.w ** 2)
            self.losses.append(loss)

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

#lambda 错误分类？  
class MultiClassSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.0001, max_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.models = {}
        self.losses = []
        self.class_to_index = {}  # 类别到索引的映射
        self.index_to_class = {}  # 索引到类别的映射

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.index_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            print(f"Training binary classifier for class {cls}...")
            binary_y = np.where(y == cls, 1, -1)
            model = SVM(self.learning_rate, self.lambda_param, self.max_iters)
            model.fit(X, binary_y)
            self.models[self.class_to_index[cls]] = model
            self.losses.append(model.losses)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, model in self.models.items():
            scores[:, idx] = np.dot(X, model.w) + model.b
        return np.array([self.index_to_class[idx] for idx in np.argmax(scores, axis=1)])



# ----------------------------- Data Loading and Preprocessing ----------------------------- #
# Load data
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# Set column names
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# Separate features and target
features = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Normalize features
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.3, random_state=42)

# ----------------------------- Multi-Class SVM Training ----------------------------- #
multi_svm = MultiClassSVM(learning_rate=0.0001, lambda_param=0.1, max_iters=10000)
multi_svm.fit(X_train, y_train)

# Predict
y_pred = multi_svm.predict(X_test)

#------------------------------- sklearn svm ---------------------------------#
# clf = svm.SVC(kernel='rbf')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#linear kernel
# Accuracy: 0.8889
# Precision (macro): 0.8911
# Recall (macro): 0.8887
# F1 Score (macro): 0.8895

#Gaussian kernal
# Accuracy: 0.9048
# Precision (macro): 0.9076
# Recall (macro): 0.9053
# F1 Score (macro): 0.9055
# ----------------------------- Model Evaluation ----------------------------- #
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Multi-Class SVM Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

# ----------------------------- Visualization ----------------------------- #
# 绘制损失函数随迭代次数的变化
for idx, losses in enumerate(multi_svm.losses):
    plt.plot(range(len(losses)), losses, label=f'Class {multi_svm.index_to_class[idx]}')
plt.title('Loss Function over Iterations for Each Class')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
# ----------------------------- Visualization ----------------------------- #
# def plot_results_with_loss(X, y, model, class_label):
#     """
#     Plot decision boundary for a specific class in multi-class SVM and loss curve
#     :param X: Input features (only 2D features for visualization)
#     :param y: Labels
#     :param model: Multi-class SVM model
#     :param class_label: Specific class to visualize
#     """
#     binary_model = model.models[model.class_to_index[class_label]]  # Get binary classifier for the class
#     x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
#     x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
#     xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01), np.arange(x1_min, x1_max, 0.01))
#     grid = np.c_[xx0.ravel(), xx1.ravel()]

#     # Add extra dimensions to grid to match training feature size
#     if grid.shape[1] < binary_model.w.shape[0]:
#         additional_dims = np.zeros((grid.shape[0], binary_model.w.shape[0] - grid.shape[1]))
#         grid = np.hstack((grid, additional_dims))

#     # Predict for the grid points
#     grid_predictions = binary_model.predict(grid)
#     grid_predictions = grid_predictions.reshape(xx0.shape)

#     # Create subplots
#     fig, axs = plt.subplots(1, 2, figsize=(16, 6))

#     # Subplot 1: Decision Boundary
#     axs[0].contourf(xx0, xx1, grid_predictions, alpha=0.8, cmap=plt.cm.coolwarm)
#     axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
#     axs[0].set_title(f"Decision Boundary for Class {class_label}", fontsize=14)
#     axs[0].set_xlabel("Feature 1", fontsize=12)
#     axs[0].set_ylabel("Feature 2", fontsize=12)

#     # Subplot 2: Loss Curve
#     axs[1].plot(range(len(binary_model.losses)), binary_model.losses, label='Training Loss', color='blue')
#     axs[1].set_title("Loss Function Over Iterations", fontsize=14)
#     axs[1].set_xlabel("Iterations", fontsize=12)
#     axs[1].set_ylabel("Loss Value", fontsize=12)
#     axs[1].legend()
#     axs[1].grid(True)

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()

# #Visualize decision boundary and loss for class 1
# plot_results_with_loss(X_test[:, :2], y_test, multi_svm, class_label=1)

