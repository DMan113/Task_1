from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, train_loader, epochs=5):
        pass

    @abstractmethod
    def predict(self, test_loader):
        pass