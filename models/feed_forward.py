import torch
import torch.nn as nn
import torch.optim as optim
from interfaces.mnist_classifier_interface import MnistClassifierInterface


class FeedForwardNN(nn.Module):
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
        return self.model(x)


class FeedForwardMnist(MnistClassifierInterface):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN().to(self.device)
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

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
            return avg_loss

    def predict(self, input_data):
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
