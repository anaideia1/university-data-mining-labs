import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k, x_label_):
        self.x_label = x_label_
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for instance in X:
            distances = self._compute_distances(instance)
            k_nearest_labels = self._get_k_nearest_labels(distances)
            majority_vote = self._get_majority_vote(k_nearest_labels)
            print(
                f'For instance {instance} we have {self.k} nearest labels: {k_nearest_labels}')
            print(f'So predicted answer is {majority_vote}')
            predictions.append(majority_vote)
        return np.array(predictions)

    def _compute_distances(self, instance):
        distances = np.sqrt(np.sum((self.X_train - instance) ** 2, axis=1))
        return distances

    def _get_k_nearest_labels(self, distances):
        sorted_indices = np.argsort(distances)
        k_nearest_labels = self.y_train[sorted_indices[:self.k]]
        return k_nearest_labels

    def _get_majority_vote(self, k_nearest_labels):
        label_counts = Counter(k_nearest_labels)
        majority_vote = label_counts.most_common(1)[0][0]
        return majority_vote


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

k = 5
classifier = KNNClassifier(k, x_label)
classifier.fit(X_train, y_train)

# trying our text value
new_instance = np.array(x_test)

prediction = classifier.predict(new_instance)
