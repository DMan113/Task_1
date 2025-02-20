import torch
import torch.nn as nn
import torch.optim as optim
from interfaces.mnist_classifier_interface import MnistClassifierInterface


class FeedForwardNN(nn.Module):
    """
    A simple feedforward neural network for MNIST classification.

    The network consists of:
    - A Flatten layer to convert 2D images into 1D vectors
    - A Linear layer with 128 neurons and ReLU activation
    - A Dropout layer to prevent overfitting
    - A final Linear layer with 10 output neurons (one per digit class)
    """

    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)


class FeedForwardMnist(MnistClassifierInterface):
    """
    A classifier wrapper for the FeedForwardNN model, implementing training and prediction.

    Attributes:
    - device: The device (CPU/GPU) on which the model runs
    - model: The neural network model
    - criterion: Loss function (CrossEntropyLoss)
    - optimizer: Adam optimizer with a learning rate of 0.001
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=5):
        """
        Trains the model using the provided training data loader.

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training data.
            epochs (int): Number of epochs for training (default: 5).

        Returns:
            float: The average loss over the last epoch.
        """
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

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
            return avg_loss

    def predict(self, input_data):
        """
        Predicts labels for the given input data.

        Args:
            input_data (torch.Tensor or torch.utils.data.DataLoader): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(input_data, '__iter__') and not isinstance(input_data, torch.Tensor):
                predictions = []
                for data, _ in input_data:
                    data = data.to(self.device)
                    outputs = self.model(data)
                    pred = outputs.argmax(dim=1)
                    predictions.extend(pred.cpu().numpy())
                return predictions
            else:
                if not isinstance(input_data, torch.Tensor):
                    input_data = torch.tensor(input_data, dtype=torch.float32)
                input_data = input_data.to(self.device)
                outputs = self.model(input_data)
                return outputs.argmax(dim=1).cpu().numpy()
