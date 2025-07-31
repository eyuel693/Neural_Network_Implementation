import numpy as np

class CategoricalCrossEntropy:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def compute(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true