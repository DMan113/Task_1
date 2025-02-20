import torch
import torch.nn as nn
import torch.optim as optim
from interfaces.mnist_classifier_interface import MnistClassifierInterface


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for MNIST classification.

    The architecture consists of:
    - Two convolutional layers with ReLU activations and max pooling.
    - A fully connected layer followed by another ReLU activation.
    - An output layer with 10 units (for the 10 MNIST digit classes).
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        return self.model(x)


class ConvNetMnist(MnistClassifierInterface):
    """
    MNIST Classifier using a Convolutional Neural Network (CNN).

    Implements the MnistClassifierInterface with training and prediction methods.
    """

    def __init__(self):
        """
        Initializes the CNN model, loss function, and optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=5):
        """
        Trains the CNN model on the given training data.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader containing training data.
            epochs (int, optional): Number of training epochs. Defaults to 5.
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

            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    def predict(self, test_loader):
        """
        Predicts labels for the given test data.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader containing test data.

        Returns:
            list: Predicted labels for the test set.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                pred = outputs.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        return predictions
