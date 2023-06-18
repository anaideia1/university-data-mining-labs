import numpy as np


class NaiveBayesClassifier:
    def __init__(self, x_label_):
        self.x_label = x_label_
        self.classes = None
        self.class_priors = None
        self.feature_probabilities = None

    def fit(self, X_train, y_train):
        self.classes, class_counts = np.unique(y_train, return_counts=True)
        self.class_priors = class_counts / len(y_train)

        self.feature_probabilities = []
        for feature_index in range(X_train.shape[1]):
            feature_values = X_train[:, feature_index]
            unique_values = np.unique(feature_values)

            feature_prob = []
            for class_label in self.classes:
                class_indices = np.where(y_train == class_label)[0]
                class_feature_values = feature_values[class_indices]
                value_counts = np.bincount(class_feature_values)

                # Laplace smoothing
                probabilities = (value_counts + 1) / (len(class_feature_values) + len(unique_values))
                feature_prob.append(probabilities)
                print('--------------------------------------------')
                print(f'Checking feature {self.x_label[feature_index]} with S={class_label}.')
                print(f'This feature has {value_counts[0]} zero-values and {value_counts[1]} one-values.')
                print('Probabilities after smoothing:', probabilities)

            self.feature_probabilities.append(feature_prob)

    def predict(self, X_test):
        print('\n\n\n--------------------------------------------')
        print('Test data:', X_test)

        predictions = []
        for instance in X_test:
            class_scores = []

            for class_index, class_label in enumerate(self.classes):
                class_score = np.log(self.class_priors[class_index])

                for feature_index, feature_value in enumerate(instance):
                    probabilities = self.feature_probabilities[feature_index][class_index]
                    feature_score = np.log(probabilities[feature_value])
                    class_score += feature_score

                class_scores.append(class_score)
                print(f'Class score for {class_label} is {class_score}')
            print('Resulting class scores', class_scores)

            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)

        print("Predictions:", predictions)
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
classifier = NaiveBayesClassifier(x_label)
classifier.fit(X, y)

# trying our text value
new_instance = np.array(x_test)

prediction = classifier.predict(new_instance)
