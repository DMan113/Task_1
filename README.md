# MNIST Classifier Project

This project implements three different classification models for the MNIST dataset using Object-Oriented Programming principles in Python. The implementation includes Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network classifiers under a unified interface.

## Project Structure

```
/test_task_1
│── /models
│   │── __init__.py
│   │── base.py
│   │── cnn.py
│   │── feed_forward.py
│   │── random_forest.py
│── /interfaces
│   │── mnist_classifier_interface.py
│── /utils
│   │── __init__.py
│   │── dataset.py
│   │── metrics.py
│   │── visualization.py
│── classifier.py
│── main.py
│── README.md
```

## Features

- Three different classification models:
  - Random Forest Classifier
  - Feed-Forward Neural Network
  - Convolutional Neural Network
- Unified interface for all models
- Automated MNIST dataset downloading and preprocessing
- Model training and evaluation capabilities
- Easy model switching through a single parameter

## Requirements

```bash
pytorch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.0
numpy==1.24.3
```

## Installation

1. Create a new conda environment:
```bash
conda create -n ml_env python=3.8
conda activate ml_env
```

2. Install required packages:
```bash
conda install pytorch torchvision -c pytorch
conda install scikit-learn numpy
```

## Usage

Basic usage example:

```python
from classifier import MnistClassifier
from utils.dataset import MnistDataset

# Get MNIST dataset
mnist_dataset = MnistDataset(batch_size=32)
train_loader, test_loader = mnist_dataset.get_data_loaders()

# Create and train model
# algorithm can be 'cnn', 'rf', or 'nn'
classifier = MnistClassifier(algorithm='nn')
classifier.train(train_loader, epochs=5)

# Make predictions
predictions = classifier.predict(test_loader)

# Evaluate model
accuracy = classifier.evaluate(test_loader)
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Model Details

### Random Forest
- Uses scikit-learn's RandomForestClassifier
- Default 100 estimators
- Suitable for smaller datasets

### Feed-Forward Neural Network
- Two-layer neural network
- ReLU activation
- Dropout for regularization
- Adam optimizer

### Convolutional Neural Network
- Two convolutional layers
- Max pooling
- ReLU activation
- Fully connected layers
- Adam optimizer

## Performance

Typical performance metrics on MNIST test set:
- Feed-Forward NN: ~97-98% accuracy
- CNN: ~98-99% accuracy
- Random Forest: ~96-97% accuracy

## Project Structure Details

- `interfaces/mnist_classifier_interface.py`: Defines the abstract interface for all classifiers
- `models/`: Contains implementation of all three classifiers
- `utils/`: Contains helper functions for data loading and processing
- `classifier.py`: Main classifier class that provides a unified interface
- `main.py`: Example usage script

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.