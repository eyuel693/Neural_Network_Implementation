import numpy as np
from model import NeuralNetwork
from optimizers import Adam
from DatasetSplitter import DatasetSplitter
from plot_result import plot_digits, plot_accuracy, plot_predictions_vs_true

splitter = DatasetSplitter(csv_path="data/train.csv", target_column="label")
splitter.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

train_data, val_data, test_data = splitter.get_data()
X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data

print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")


X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

optimizer = Adam(alpha=0.001)
input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))
layer_sizes = [input_size, 128, 64, num_classes]
model = NeuralNetwork(layer_sizes=layer_sizes, optimizer=optimizer)


train_losses, val_losses, train_accs, val_accs = model.train(
    X_train, y_train, X_val=X_val, y_val=y_val, epochs=50, batch_size=64, patience=10
)


plot_digits(X_train, y_train, model, num_images=10, dataset_name="Train")
plot_digits(X_val, y_val, model, num_images=10, dataset_name="Validation")
plot_accuracy(train_accs, val_accs)


y_test_pred = model.predict(X_test)
test_acc = model.accuracy(y_test_pred, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

plot_predictions_vs_true(y_test, y_test_pred, num_samples=50)
