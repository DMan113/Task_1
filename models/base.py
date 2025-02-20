import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BaseModel:
    """
    Base class for machine learning models using PyTorch.

    Attributes:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
        criterion (nn.CrossEntropyLoss): The loss function for classification.
    """

    def __init__(self, model, optimizer):
        """
        Initializes the model, optimizer, and loss function.

        Args:
            model (nn.Module): The PyTorch model to be used.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """
        Trains the model using the given dataset.

        Args:
            X_train (Tensor): The training input data.
            y_train (Tensor): The corresponding training labels.
            epochs (int, optional): The number of training epochs (default: 5).
            batch_size (int, optional): The batch size for training (default: 32).
        """
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

    def predict(self, X_test, batch_size=32):
        """
        Generates predictions using the trained model.

        Args:
            X_test (Tensor): The test input data.
            batch_size (int, optional): The batch size for inference (default: 32).

        Returns:
            list: Predicted labels for the input data.
        """
        dataset = TensorDataset(X_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data[0].to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        return predictions

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluates the model performance on the test dataset.

        Args:
            X_test (Tensor): The test input data.
            y_test (Tensor): The ground truth labels.
            batch_size (int, optional): The batch size for evaluation (default: 32).

        Returns:
            float: Accuracy of the model on the test dataset.
        """
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy
