import numpy as np


class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree(instance, self.tree) for instance in
                       X]
        return np.array(predictions)

    def _build_tree(self, X, y):
        # Base cases for recursion
        if len(np.unique(y)) == 1:  # All instances belong to the same class
            return {'label': y[0]}
        if X.shape[1] == 0:  # No more features to split on
            return {'label': np.bincount(y).argmax()}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Create a decision node for the best split
        decision_node = {'feature': best_feature, 'threshold': best_threshold}

        # Split the data based on the best feature and threshold
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Recursively build the left and right subtrees
        decision_node['left'] = self._build_tree(X[left_indices],
                                                 y[left_indices])
        decision_node['right'] = self._build_tree(X[right_indices],
                                                  y[right_indices])

        return decision_node

    def _find_best_split(self, X, y):
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            feature_values = np.unique(X[:, feature])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                left_gini = self._compute_gini(y[left_indices])
                right_gini = self._compute_gini(y[right_indices])

                weighted_gini = (left_gini * sum(
                    left_indices) + right_gini * sum(right_indices)) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _compute_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _traverse_tree(self, instance, node):
        if 'label' in node:
            return node['label']

        feature = node['feature']
        threshold = node['threshold']

        if instance[feature] <= threshold:
            return self._traverse_tree(instance, node['left'])
        else:
            return self._traverse_tree(instance, node['right'])


# Example (f variant)
# Q1 Q2 Q3 Q4  S
#  0  0  0  0  1
#  0  0  0  1  1
#  0  0  1  0  1
#  0  0  1  1  1
#  0  1  0  0  0
#  0  1  0  1  0
#  0  1  1  0  0
#  0  1  1  1  1
#  1  0  0  0  1
#  1  0  1  0  0


x_label = ['Q1', 'Q2', 'Q3', 'Q4']
X_train = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [0, 1, 0, 0],
                    [0, 1, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 1, 0]])
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
x_test = [[1, 1, 1, 1]]

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

new_instances = np.array(x_test)

predictions = classifier.predict(new_instances)
print("Predictions:", predictions)
