import numpy as np
import torch
from classifier import MnistClassifier
from utils.dataset import MnistDataset
from utils.metrics import ClassificationMetrics
from utils.visualization import ModelVisualizer

# Initialize MNIST dataset with batch size 32
mnist_dataset = MnistDataset(batch_size=32)
train_loader, test_loader = mnist_dataset.get_data_loaders()

# Create an MNIST classifier using a feedforward neural network ('nn')
classifier = MnistClassifier(algorithm='nn')

# Lists to store training loss and accuracy per epoch
losses = []
accuracies = []

# Train the model for 5 epochs and evaluate accuracy after each epoch
for epoch in range(5):
    loss = classifier.train(train_loader, epochs=1)
    accuracy = classifier.evaluate(test_loader)
    losses.append(loss)
    accuracies.append(accuracy)
    print(f"Accuracy: {accuracy:.2f}%")

# Lists to store predictions, ground truth labels, and images
all_predictions = []
all_true_labels = []
all_images = []

# Iterate over the test dataset and collect predictions
for data, target in test_loader:
    batch_predictions = classifier.predict(data)

    all_predictions.extend(batch_predictions)
    all_true_labels.extend(target.numpy())
    all_images.extend(data.numpy())

# Convert lists to NumPy arrays for further analysis
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)
all_images = np.array(all_images)

# Calculate classification metrics (accuracy, precision, recall, etc.)
metrics = ClassificationMetrics.calculate_metrics(all_true_labels, all_predictions)
ClassificationMetrics.print_metrics(metrics)

# Visualize training performance and model results
ModelVisualizer.plot_training_history(losses, accuracies)  # Loss & accuracy over epochs
ModelVisualizer.plot_confusion_matrix(metrics['confusion_matrix'])  # Confusion matrix
ModelVisualizer.plot_predictions(all_images, all_predictions, all_true_labels)  # Sample predictions
ModelVisualizer.plot_misclassified(all_images, all_predictions, all_true_labels)  # Misclassified samples
