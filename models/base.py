import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BaseModel:
    def __init__(self, model, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs=5, batch_size=32):
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
