import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interfaces.mnist_classifier_interface import MnistClassifierInterface
import torch


class RandomForestMnist(MnistClassifierInterface):
    """
    A Random Forest-based classifier for the MNIST dataset.
    This class implements the MnistClassifierInterface and uses
    the RandomForestClassifier from scikit-learn to perform image classification.

    Attributes:
        model (RandomForestClassifier): The Random Forest model used for classification.
        device (str): The computing device, set to 'cpu' as RandomForestClassifier does not support GPU acceleration.
    """

    def __init__(self):
        """
        Initializes the RandomForestMnist classifier.
        The model is configured with 100 estimators and a fixed random state for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.device = 'cpu'

    def train(self, train_loader, epochs=5):
        """
        Trains the Random Forest classifier using data from the train_loader.

        Args:
            train_loader (DataLoader): A PyTorch DataLoader containing the training dataset.
            epochs (int, optional): The number of epochs to train for (default is 5).
                                    This parameter is not used in RandomForest but is included for interface consistency.

        The function extracts image data and labels from the train_loader, reshapes them into
        a format suitable for scikit-learn, and fits the Random Forest model.
        """
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
        """
        Predicts the class labels for the given test dataset.

        Args:
            test_loader (DataLoader): A PyTorch DataLoader containing the test dataset.

        Returns:
            numpy.ndarray: An array of predicted class labels for the test dataset.

        The function extracts and reshapes the image data from test_loader
        before using the trained Random Forest model to make predictions.
        """
        X_test_list = []
        for data, _ in test_loader:
            X_test_list.append(data.numpy().reshape(data.shape[0], -1))

        X_test = np.concatenate(X_test_list)
        return self.model.predict(X_test)
