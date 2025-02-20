import torch
import torch.nn as nn
import torch.optim as optim
from interfaces.mnist_classifier_interface import MnistClassifierInterface


class CNN(nn.Module):
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
        return self.model(x)


class ConvNetMnist(MnistClassifierInterface):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=5):
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
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                pred = outputs.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        return predictions