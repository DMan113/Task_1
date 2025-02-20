import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision.utils import make_grid


class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
        """
        Plot confusion matrix using seaborn
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_training_history(losses, accuracies=None):
        """
        Plot training history (loss and accuracy)
        """
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if accuracies is not None:
            plt.subplot(1, 2, 2)
            plt.plot(accuracies)
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_predictions(images, predictions, true_labels, num_images=10):
        """
        Plot images with their predictions and true labels
        """
        plt.figure(figsize=(20, 4))
        for i in range(min(num_images, len(images))):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            plt.title(f'Pred: {predictions[i]}\nTrue: {true_labels[i]}',
                      color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_misclassified(images, predictions, true_labels, num_images=10):
        """
        Plot misclassified images
        """
        misclassified_indices = np.where(predictions != true_labels)[0]
        if len(misclassified_indices) == 0:
            print("No misclassified images found!")
            return

        plt.figure(figsize=(20, 4))
        for i in range(min(num_images, len(misclassified_indices))):
            idx = misclassified_indices[i]
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[idx].squeeze(), cmap='gray')
            plt.title(f'Pred: {predictions[idx]}\nTrue: {true_labels[idx]}',
                      color='red')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
