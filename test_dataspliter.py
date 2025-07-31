import pandas as pd
import matplotlib.pyplot as plt

from DatasetSplitter  import DatasetSplitter

splitter = DatasetSplitter(csv_path="train.csv", target_column="label")
splitter.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

train_data, val_data, test_data = splitter.get_data()

X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Testing data shape:", X_test.shape)
