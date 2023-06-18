import numpy as np
from collections import Counter


class OneRuleClassifier:
    def __init__(self, x_label_):
        self.x_label = x_label_
        self.best_rule = None

    def fit(self, X_train, y_train):
        best_accuracy = 0.0

        for feature_index in range(X_train.shape[1]):
            feature_values = X_train[:, feature_index]
            unique_values = np.unique(feature_values)

            for value in unique_values:
                rule = (feature_index, value)
                print('--------------------------------------------')
                print(f'Checking rule for feature {self.x_label[feature_index]} with value {value}.')
                class_counts = Counter(y_train[feature_values == value])
                most_common_class = class_counts.most_common(1)[0][0]
                correct_predictions = class_counts[most_common_class]
                # Calculate the accuracy of the rule
                accuracy = correct_predictions / len(y_train[feature_values == value])
                print(f'Current rule has accuracy {accuracy}.\nBest accuracy for now is {best_accuracy}')
                if accuracy > best_accuracy:
                    self.best_rule = rule
                    best_accuracy = accuracy
                    print('So we setting this rule as our new best rule.')
                else:
                    print('This rule has lower accuracy, so we skipping it.')

        print('========================================')
        print(f'Resulting best rule:\n\"If {self.x_label[self.best_rule[0]]} feature is equal to {self.best_rule[1]} then S=1\" with accuracy {best_accuracy}')

    def predict(self, X):
        if self.best_rule is None:
            raise Exception("Classifier not trained")

        feature_index, value = self.best_rule
        predictions = np.zeros(X.shape[0])
        predictions[X[:, feature_index] == value] = 1

        return predictions

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
X = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 1, 0]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
x_test = [[1, 1, 1, 1]]

# fitting our classifier
classifier = OneRuleClassifier(x_label)
classifier.fit(X, y)

# trying our text value
new_instance = np.array(x_test)

prediction = classifier.predict(new_instance)
print('\n\n\n--------------------------------------------')
print('Test data:', x_test)
print("Prediction:", prediction)
