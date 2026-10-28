import numpy as np


class LogisticRegression:
    def fit(self, X, Y, learning_rate, L2_par = 0, epochs = 1000):
        self.samples, self.features = X.shape
        self.theta = np.zeros((self.features, 1))
        self.b = 0

        for e in range(epochs):
            z = X @ self.theta + self.b
            Y_prediction = self.sigmoid(z)
            self.b -= learning_rate * np.mean(Y_prediction - Y)
            self.theta -= learning_rate * ((X.T @ (Y_prediction - Y) + L2_par * self.theta) / self.samples) 

        return self.theta, self.b
    
    def sigmoid(self, x):
        return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
        
    def predict(self, X):
        prediction = self.sigmoid(X @ self.theta + self.b)
        result = np.where(prediction >= 0.5, 1, 0)
        return result
    
    def accuracy(self, Y, Y_prediction):
        return (Y.shape[0] - np.count_nonzero(Y-Y_prediction))/Y.shape[0]

    def error(self, Y, Y_prediction):
        return np.count_nonzero(Y-Y_prediction)/Y.shape[0]
    

        
    