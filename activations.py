import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad_output):
        raise NotImplementedError()

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0).astype(float)

class Softmax(Activation):
    def forward(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output

class Sigmoid(Activation):
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)