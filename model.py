import numpy as np
from layers import DenseLayer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy

class NeuralNetwork:
    def __init__(self, layer_sizes, optimizer, activation_hidden=ReLU, activation_output=Softmax, loss_fn=CategoricalCrossEntropy):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            act = activation_hidden() if i < len(layer_sizes) - 2 else activation_output()
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], act, optimizer, layer_index=i))
        self.loss_fn = loss_fn()

    def one_hot(self, y):
        if y.ndim > 1:
            return y
        num_classes = np.max(y) + 1
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_pred, y_true):
        grad = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X_train, y_train, X_val=None, y_val=None, *, epochs, batch_size, patience=None):
        y_one_hot = self.one_hot(y_train)
        if X_val is not None and y_val is not None:
            y_val_one_hot = self.one_hot(y_val)
        else:
            y_val_one_hot = None

        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            total_loss = 0.0
            num_batches = 0
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[indices[i:i+batch_size]]
                y_batch = y_one_hot[indices[i:i+batch_size]]
                if len(X_batch) < 1:
                    continue
                output = self.forward(X_batch)
                total_loss += self.loss_fn.compute(output, y_batch)
                num_batches += 1
                self.backward(output, y_batch)

            avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            y_pred = self.predict(X_train)
            train_acc = self.accuracy(y_pred, y_train)
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)

            val_loss = float('inf')
            val_acc = 0.0
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_acc = self.accuracy(y_val_pred, y_val)
                val_loss = self.loss_fn.compute(self.forward(X_val), y_val_one_hot)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}", end="")
                if X_val is not None and y_val is not None:
                    print(f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                else:
                    print()

            # Early stopping
            if patience is not None and X_val is not None and y_val is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}: No improvement in validation loss for {patience} epochs")
                        # Restore best parameters
                        for layer, (W, b) in zip(self.layers, best_params):
                            layer.W = W
                            layer.b = b
                        break

        return train_losses, val_losses, train_accs, val_accs

    def accuracy(self, y_pred, y_true):
        y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        return np.mean(y_pred == y_true_labels)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)