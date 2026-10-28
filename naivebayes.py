import numpy as np


class NaiveBayes:
    def __init__(self, class_values, feature_values):
        self.feature_values = feature_values
        self.classes = class_values

    def fit(self, X, Y, laplace_smoothing_par):
        self.laplace_par = laplace_smoothing_par
        self.samples, self.features = X.shape

        self.probabilites = np.zeros((len(self.classes), self.features, len(self.feature_values)))
        self.y_apriori = np.zeros(len(self.classes))

        for c in range(len(self.classes)):
            # rows of that particular class from samples
            X_subset = X[np.where(Y==self.classes[c])[0]] 
            Y_subset = Y[np.where(Y==self.classes[c])[0]]
            self.y_apriori[c] = (len(Y_subset) + self.laplace_par)/(self.samples + self.laplace_par * len(self.classes))

            for f in range(self.features):
                for v in range(len(self.feature_values)):
                    self.probabilites[c][f][v] = (
                        (np.sum(X_subset[:,f] == self.feature_values[v]) + self.laplace_par)/
                        (self.laplace_par * len(self.feature_values) + X_subset.shape[0])
                    )


    def class_probability(self, x, c, log):
        result = self.y_apriori[c]

        if not log:
            for i in range(self.features):
                result = result * self.probabilites[c][i][np.where(self.feature_values == x[i])[0][0]]
        else:
            result = np.log(result)
            for i in range(self.features):
                result = result + np.log(self.probabilites[c][i][np.where(self.feature_values == x[i])[0][0]])

        return result

    # log determines whether we use the sum or the product of probabilities
    def predict(self, X, log = False):
        probabilities = np.zeros(len(self.classes))

        result = []

        for x in X:
            for c in range(len(self.classes)):
                probabilities[c] = self.class_probability(x, c, log)

            result.append(self.classes[np.argmax(probabilities)])

        return np.array(result).reshape(-1,1)
    
    def accuracy(self, Y, Y_prediction):
        return (Y.shape[0] - np.count_nonzero(Y-Y_prediction))/Y.shape[0]

    def error(self, Y, Y_prediction):
        return np.count_nonzero(Y-Y_prediction)/Y.shape[0]
        


       