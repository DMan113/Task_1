from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MnistDataset:
    """
    A class for handling the MNIST dataset with preprocessing and DataLoader setup.

    Attributes:
        batch_size (int): The number of samples per batch.
        transform (transforms.Compose): Transformations applied to the dataset.
    """

    def __init__(self, batch_size=32):
        """
        Initializes the MnistDataset with the given batch size and transformation.

        Args:
            batch_size (int, optional): Number of samples per batch. Default is 32.
        """
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_data_loaders(self):
        """
        Loads and returns the training and test data loaders for the MNIST dataset.

        Returns:
            tuple: (train_loader, test_loader), where each is a DataLoader for the dataset.
        """
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
