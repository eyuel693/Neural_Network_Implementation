import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation, optimizer, layer_index, clip_norm=1.0):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)  
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.optimizer = optimizer
        self.layer_index = layer_index  
        self.clip_norm = clip_norm
        self.last_input = None
        self.last_z = None

    def forward(self, X):
        self.last_input = X
        self.last_z = np.dot(X, self.W.T) + self.b
        return self.activation.forward(self.last_z)

    def backward(self, grad_output):
        try:
            if grad_output.shape[0] != self.last_input.shape[0]:
                raise ValueError(f"Batch size mismatch: grad_output ({grad_output.shape[0]}) "
                                f"vs last_input ({self.last_input.shape[0]})")
            if grad_output.shape[1] != self.last_z.shape[1]:
                raise ValueError(f"Output size mismatch: grad_output ({grad_output.shape[1]}) "
                                f"vs last_z ({self.last_z.shape[1]})")

            grad_activation = self.activation.backward(grad_output)

            if grad_activation.shape != self.last_z.shape:
                raise ValueError(f"Gradient activation shape mismatch: got {grad_activation.shape}, "
                                f"expected {self.last_z.shape}")

            grad_activation = np.clip(grad_activation, -self.clip_norm, self.clip_norm)

            dW = np.dot(grad_activation.T, self.last_input) / max(1, self.last_input.shape[0])
            db = np.mean(grad_activation, axis=0, keepdims=True)

            grad_input = np.dot(grad_activation, self.W)

            self.W = self.optimizer.update(self.W, dW, key=f"W_layer_{self.layer_index}")
            self.b = self.optimizer.update(self.b, db, key=f"b_layer_{self.layer_index}")

            if grad_input.shape != self.last_input.shape:
                raise ValueError(f"Gradient input shape mismatch: got {grad_input.shape}, "
                                f"expected {self.last_input.shape}")

            return grad_input

        except Exception as e:
            raise RuntimeError(f"Backward pass failed: {str(e)}")