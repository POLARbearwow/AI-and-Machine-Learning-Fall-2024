import numpy as np
from graphviz import Digraph
import os


class DecisionTree:
    # Create an internal class called Node
    class Node:
        def __init__(self) -> None:
            # When a node is a leaf node, then we use the output label as the value
            self.value = None
            # When a node is an internal node, then we use feature_index on it
            self.feature_index = None
            # Use a dictionary called children to indicate the children nodes, which contain {feature_value: node, }
            self.children = {}

        def __str__(self) -> str:
            if self.children:
                s = f'Internal node <{self.feature_index}>:\n'
                for fv, node in self.children.items():
                    ss = f'[{fv}]-> {node}'
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = f'Leaf node ({self.value})'

            return s

    def __init__(self, gain_threshold=1e-2) -> None:
        # Set a threshold for information gain
        self.gain_threshold = gain_threshold

    def _entropy(self, y):
        # Compute entropy of output -sum(p(Y=y)log2(p(Y=y))), which is a scalar
        count_y = np.bincount(y)
        prob_y = count_y[np.nonzero(count_y)] / y.size
        entropy_y = -np.sum(prob_y * np.log2(prob_y))
        return entropy_y

    def _conditional_entropy(self, feature, y):
        # Compute the conditional entropy H(Y|feature)
        feature_values = np.unique(feature)
        h = 0.
        for v in feature_values:
            y_sub = y[feature == v]
            prob_y_sub = y_sub.size / y.size
            h += prob_y_sub * self._entropy(y_sub)
        return h

    def _information_gain(self, feature, y):
        return self._entropy(y) - self._conditional_entropy(feature, y)

    def _select_feature(self, X, y, features_list):
        # Select the feature with the largest information gain
        if features_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, features_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index
        return None

    def _build_tree(self, X, y, features_list):
        # Build a decision tree recursively
        node = DecisionTree.Node()
        labels_count = np.bincount(y)
        node.value = np.argmax(labels_count)

        # Check whether the labels are the same
        if np.count_nonzero(labels_count) != 1:
            # Select the feature with the largest information gain
            index = self._select_feature(X, y, features_list)
            if index is not None:
                # Remove this feature from the features list
                node.feature_index = features_list.pop(index)
                # Divide the training set according to this selected feature
                feature_values = np.unique(X[:, node.feature_index])
                for v in feature_values:
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]
                    node.children[v] = self._build_tree(X_sub, y_sub, features_list.copy())
        return node

    def train(self, X_train, y_train):
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))

    def _predict_one(self, x):
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                break
            node = child
        return node.value

    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)

    def __str__(self):
        return str(self.tree_) if hasattr(self, 'tree_') else ''


class DecisionTreePlotter:
    def __init__(self, tree, feature_names=None, label_names=None) -> None:
        self.tree = tree
        self.feature_names = feature_names
        self.label_names = label_names
        self.graph = Digraph('Decision Tree')

    def _build(self, dt_node):
        if dt_node.children:
            d = self.feature_names[dt_node.feature_index]
            label = d['name'] if self.feature_names else str(dt_node.feature_index)
            self.graph.node(str(id(dt_node)), label=label, shape='box')

            for feature_value, dt_child in dt_node.children.items():
                self._build(dt_child)
                d_value = d.get('value_names')
                edge_label = d_value[feature_value] if d_value else str(feature_value)
                self.graph.edge(str(id(dt_node)), str(id(dt_child)), label=edge_label, fontsize='10')
        else:
            # Check bounds for label_names
            if 0 <= dt_node.value < len(self.label_names):
                label = self.label_names[dt_node.value]
            else:
                label = f"Unknown ({dt_node.value})"
            self.graph.node(str(id(dt_node)), label=label, shape='ellipse')

    def plot(self):
        self._build(self.tree)
        self.graph.view()


if __name__ == '__main__':
    # Step 1: Load data
    print("Current Working Directory:", os.getcwd())
    data = np.loadtxt('hw7/HW_DT/dataset/lenses/lenses.data', dtype=int)
    X = data[:, 1:-1]
    y = data[:, -1]

    # Step 2: Split data into train and test sets
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Step 3: Train decision tree
    dt = DecisionTree()
    dt.train(X_train, y_train)

    # Step 4: Evaluate decision tree on test set
    y_pred = dt.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy on test set: {accuracy:.2f}")

    # Step 5: Visualize decision tree
    feature_names = [
        {'name': 'Age'},
        {'name': 'Prescription'},
        {'name': 'Astigmatism'},
        {'name': 'Tear Rate'}
    ]
    label_names = ['Soft Lens', 'Hard Lens', 'No Lens']
    plotter = DecisionTreePlotter(dt.tree_, feature_names=feature_names, label_names=label_names)
    plotter.plot()
