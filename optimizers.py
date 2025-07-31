import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, clip_norm=1.0):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if clip_norm <= 0:
            raise ValueError("Clip norm must be positive")
            
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm

    def update(self, param, grad, key=None):
        if param.shape != grad.shape:
            raise ValueError(f"Parameter and gradient shape mismatch: {param.shape} vs {grad.shape}")
            
        grad = np.clip(grad, -self.clip_norm, self.clip_norm)
        return param - self.learning_rate * grad

class Adam:
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clip_norm=1.0):
        if alpha <= 0 or beta_1 <= 0 or beta_2 <= 0 or epsilon <= 0 or clip_norm <= 0:
            raise ValueError("All hyperparameters must be positive")
        if beta_1 >= 1 or beta_2 >= 1:
            raise ValueError("Beta parameters must be less than 1")
            
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param, grad, key):
        if param.shape != grad.shape:
            raise ValueError(f"Parameter and gradient shape mismatch: {param.shape} vs {grad.shape}")
        if key is None:
            raise ValueError("Key must be provided for parameter tracking")

        grad = np.clip(grad, -self.clip_norm, self.clip_norm)

        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
            self.t[key] = 0

        self.t[key] += 1
        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta_1 ** self.t[key])
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t[key])

        update = self.alpha * m_hat / (np.sqrt(np.maximum(v_hat, 0)) + self.epsilon)
        return param - update