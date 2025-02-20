import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interfaces.mnist_classifier_interface import MnistClassifierInterface
import torch


class RandomForestMnist(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.device = 'cpu'

    def train(self, train_loader, epochs=5):
        X_train_list = []
        y_train_list = []
        for data, target in train_loader:
            X_train_list.append(data.numpy().reshape(data.shape[0], -1))
            y_train_list.append(target.numpy())

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        self.model.fit(X_train, y_train)
        print("Random Forest training completed")

    def predict(self, test_loader):
        X_test_list = []
        for data, _ in test_loader:
            X_test_list.append(data.numpy().reshape(data.shape[0], -1))

        X_test = np.concatenate(X_test_list)
        return self.model.predict(X_test)
