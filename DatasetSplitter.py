import numpy as np
import pandas as pd

class DatasetSplitter:
    def __init__(self, X=None, y=None, csv_path=None, target_column=None):
        if csv_path:
            data = pd.read_csv(csv_path)
            assert target_column in data.columns, "Target column not found in CSV"
            self.X = data.drop(columns=[target_column]).values
            self.y = data[target_column].values
        else:
            assert X is not None and y is not None, "Either data arrays or csv_path must be provided"
            self.X = X
            self.y = y
        
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def shuffle_data(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]

    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        self.shuffle_data()
        total = len(self.X)
        
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        X_train, y_train = self.X[:train_end], self.y[:train_end]
        X_val, y_val = self.X[train_end:val_end], self.y[train_end:val_end]
        X_test, y_test = self.X[val_end:], self.y[val_end:]

        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data
