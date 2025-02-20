from models.cnn import ConvNetMnist
from models.feed_forward import FeedForwardMnist
from models.random_forest import RandomForestMnist


class MnistClassifier:
    """
    A classifier for MNIST dataset supporting multiple algorithms: CNN, Feed-Forward NN, and Random Forest.
    """

    def __init__(self, algorithm='cnn'):
        """
        Initializes the classifier with the specified algorithm.

        :param algorithm: The algorithm to use ('cnn', 'rf', 'nn').
        :raises ValueError: If the specified algorithm is not supported.
        """
        if algorithm == 'cnn':
            self.classifier = ConvNetMnist()
        elif algorithm == 'rf':
            self.classifier = RandomForestMnist()
        elif algorithm == 'nn':
            self.classifier = FeedForwardMnist()
        else:
            raise ValueError("Algorithm must be one of: 'cnn', 'rf', 'nn'")

    def train(self, train_loader, epochs=5):
        """
        Trains the classifier on the given dataset.

        :param train_loader: DataLoader for the training dataset.
        :param epochs: Number of epochs to train.
        :return: The final training loss.
        """
        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            batches = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.classifier.device), target.to(self.classifier.device)

                if hasattr(self.classifier, 'optimizer'):
                    self.classifier.optimizer.zero_grad()
                    output = self.classifier.model(data)
                    loss = self.classifier.criterion(output, target)
                    loss.backward()
                    self.classifier.optimizer.step()
                    running_loss += loss.item()
                    batches += 1

            if hasattr(self.classifier, 'optimizer'):
                avg_loss = running_loss / batches
                print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
                epoch_losses.append(avg_loss)
            else:  # For Random Forest
                self.classifier.train(train_loader)
                print(f'Epoch {epoch + 1}, Training completed')
                epoch_losses.append(0.0)

        return epoch_losses[-1] if epoch_losses else 0.0

    def predict(self, test_loader):
        """
        Predicts labels for the given test dataset.

        :param test_loader: DataLoader for the test dataset.
        :return: Predicted labels.
        """
        return self.classifier.predict(test_loader)

    def evaluate(self, test_loader):
        """
        Evaluates the accuracy of the classifier on the test dataset.

        :param test_loader: DataLoader for the test dataset.
        :return: Accuracy as a percentage.
        """
        self.classifier.model.eval() if hasattr(self.classifier.model, 'eval') else None
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.classifier.device), target.to(self.classifier.device)
            if hasattr(self.classifier, 'model'):
                output = self.classifier.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
            else:
                pred = self.classifier.predict(data.cpu())
                correct += (pred == target.cpu().numpy()).sum()
            total += target.size(0)

        accuracy = 100. * correct / total
        return accuracy
