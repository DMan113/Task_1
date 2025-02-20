from classifier import MnistClassifier
from utils.dataset import MnistDataset
from utils.metrics import ClassificationMetrics
from utils.visualization import ModelVisualizer
import numpy as np
import torch


mnist_dataset = MnistDataset(batch_size=32)
train_loader, test_loader = mnist_dataset.get_data_loaders()

classifier = MnistClassifier(algorithm='nn')

losses = []
accuracies = []

for epoch in range(5):
    loss = classifier.train(train_loader, epochs=1)
    accuracy = classifier.evaluate(test_loader)
    losses.append(loss)
    accuracies.append(accuracy)
    print(f"Accuracy: {accuracy:.2f}%")

all_predictions = []
all_true_labels = []
all_images = []

for data, target in test_loader:
    batch_predictions = classifier.predict(data)

    all_predictions.extend(batch_predictions)
    all_true_labels.extend(target.numpy())
    all_images.extend(data.numpy())

all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)
all_images = np.array(all_images)

metrics = ClassificationMetrics.calculate_metrics(all_true_labels, all_predictions)
ClassificationMetrics.print_metrics(metrics)

ModelVisualizer.plot_training_history(losses, accuracies)
ModelVisualizer.plot_confusion_matrix(metrics['confusion_matrix'])
ModelVisualizer.plot_predictions(all_images, all_predictions, all_true_labels)
ModelVisualizer.plot_misclassified(all_images, all_predictions, all_true_labels)
