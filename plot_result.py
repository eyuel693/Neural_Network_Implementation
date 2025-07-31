import matplotlib.pyplot as plt
import numpy as np

def plot_digits(X, y, model, num_images=10, dataset_name="Train"):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        image = X[i].reshape(28, 28)
        prediction = model.predict(X[i:i+1])[0]
        label = y[i]
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title(f'Pred: {prediction}, True: {label}')
        plt.axis('off')
    plt.suptitle(f'{dataset_name} Set Predictions')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_accuracy(train_accs, val_accs):
    if len(train_accs) == 0 or len(val_accs) == 0:
        raise ValueError("Accuracy lists are empty.")

    if len(train_accs) != len(val_accs):
        raise ValueError("train_accs and val_accs must be of the same length.")

    epoch_range = range(1, len(train_accs) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_accs, label='Train Accuracy', marker='o', color='blue')
    plt.plot(epoch_range, val_accs, label='Validation Accuracy', marker='s', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xticks(epoch_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_true(y_true, y_pred, num_samples=50):
    plt.figure(figsize=(12, 5))
    plt.plot(range(num_samples), y_true[:num_samples], 'bo', label='True Labels')
    plt.plot(range(num_samples), y_pred[:num_samples], 'rx', label='Predicted Labels')
    plt.legend()
    plt.title('True vs Predicted Labels (First 50 Test Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.show()
