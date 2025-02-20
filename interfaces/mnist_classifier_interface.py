from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classifiers.

    All classifier implementations must inherit from this interface and
    implement the `train` and `predict` methods.
    """

    @abstractmethod
    def train(self, train_loader, epochs=5):
        """
        Trains the classifier using the given training data.

        :param train_loader: DataLoader object containing the training dataset.
        :param epochs: Number of training epochs (default: 5).
        """
        pass

    @abstractmethod
    def predict(self, test_loader):
        """
        Generates predictions for the given test dataset.

        :param test_loader: DataLoader object containing the test dataset.
        :return: Predicted labels or class probabilities.
        """
        pass
