from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MnistDataset:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_data_loaders(self):
        """Return train and test data loaders"""
        train_dataset = datasets.MNIST('../data',
                                       train=True,
                                       download=True,
                                       transform=self.transform)

        test_dataset = datasets.MNIST('../data',
                                      train=False,
                                      transform=self.transform)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)

        return train_loader, test_loader
